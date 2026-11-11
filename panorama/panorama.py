"""
Takes the overlapping photos from the science module camera and stitches them
into one wide panorama.

Program Layout
1. Look for distinctive spots in every photo (corners, specks, bits of texture).
2. Match those spots between photos so we know which ones overlap.
3. Use the matches to figure out where the camera was pointing for each shot.
4. Wrap every photo onto a cylinder, even out the brightness, and blend the seams.

Start reading at create_panorama(). It runs those four steps in order.
Shrunk-copy matching, stage timing, and this troubleshooting guide come from
Nate's stitch_panorama.py.

How to run it:
    python panorama.py                 # stitch captures/ into panorama.jpg
    python panorama.py --show          # same thing, then pop up a window with the result
    python panorama.py --input some_folder --output some_name.jpg

Troubleshooting (each stage prints its time, so you can see where it's slow):
- "Features found" or "Matching done" is slow -> lower WORK_PIXELS, or set a
  MAX_FEATURES cap (test it first, see the note on that knob).
- "Only X of Y photos linked up" -> those photos don't overlap enough or have too
  few spots. Retake with 30-50% overlap, raise WORK_PIXELS, or lower MIN_CONFIDENCE.
- "bundle adjustment could not line the cameras up" -> a weak or wrong link got
  through. Raise MIN_CONFIDENCE or MATCH_STRICTNESS.
- "disagree on the focal length" or "canvas too big" -> the camera estimate came
  out wrong. Usually bad overlap, something moving in the scene, or a wide lens
  bowing straight lines. For the last one set LENS_HFOV and LENS_K1.
- Slow or crashing after "Cameras aligned" -> blending is short on memory.
  Lower SEAM_PIXELS.
- Ghosting or double images -> the camera moved, not just turned, between
  shots, or something in the scene moved.
"""

import time

SCRIPT_START = time.perf_counter()  # start the clock before the slow OpenCV import
_last_mark = SCRIPT_START

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent  # so the paths work no matter where you run it from
IMAGE_FOLDER = HERE / "captures"
OUTPUT_FILE = HERE / "panorama.jpg"
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}  # checked in lowercase, so .JPG still counts

# Tuning knobs
WORK_PIXELS = 600_000     # spots are found and matched on copies shrunk to about this many pixels.
                          # The main speed knob; matching doesn't get more accurate at full size.
SIFT_CONTRAST = 0.01      # lower = finds spots on plainer surfaces (OpenCV's default is 0.04)
SIFT_EDGE_LIMIT = 20      # higher = keeps more spots that sit on edges (OpenCV's default is 10)
# Spots kept per photo, 0 = no limit. A cap keeps only the strongest spots, which
# pile up on busy areas and leave plain walls with nothing to match. At full size,
# 4000 failed every Nov 6 run. On shrunk copies 4000 passed all 6 test sets and
# saved a few seconds, but that's not enough testing to trust it yet.
MAX_FEATURES = 0
MATCH_STRICTNESS = 0.4    # a spot's best match has to beat its second best by 40%
# How sure we need to be that two photos link. OpenCV's default is 1.0, but real
# links between wide-lens Navcam frames scored 0.75-0.95. Dropping to 0.7 alone let
# a false link into captures/ (12 agreeing spots, score 0.84), so MIN_INLIERS backs it up.
MIN_CONFIDENCE = 0.7
MIN_INLIERS = 15          # a link needs at least this many spots that agree on the geometry.
                          # Real links had 27-105; the false one had 12. Tight margin, watch it.
SEQUENTIAL_MATCHING = False  # set True (or --in-order) for servo shots taken in order: only neighbours get compared
NEIGHBOUR_RANGE = 1          # how many photos ahead count as a neighbour
# In order, a false link between far-apart photos can't happen, so the bar can sit lower.
# That rescued low-res noisy captures (a real 3-4 link had 14 spots) without new false links.
IN_ORDER_MIN_CONFIDENCE = 0.5
IN_ORDER_MIN_INLIERS = 8
# With a known lens, a link between two photos from a turning camera must be a pure
# rotation. Good Navcam links stretched 3-5x (loose fit over a thin overlap); a false
# link on a noisy frame stretched 13,000x. Links stretched more than this get dropped.
MAX_LINK_STRETCH = 10
BLUR_WARNING = 0.72       # warn when a photo's edges are less than this fraction as crisp as typical.
                          # Clean test photos never went below 0.81; motion-blurred ones scored 0.52-0.63.
SEAM_PIXELS = 100_000     # seams and brightness get planned on shrunk copies about this many pixels
MAX_CANVAS_RATIO = 3.0    # give up if the canvas is bigger than this times all the photos combined,
MAX_CANVAS_MEGAPIX = 150  # or bigger than this many megapixels. A bad focal length blows past both.
BLEND_STRENGTH = 5        # how wide the seam blend is, as a percent of the panorama size
ROTATE_PHOTOS = 0         # turn every photo clockwise by 0, 90, 180 or 270 deg first (match the camera mount)
TRIM_DARK_BORDERS = True  # cut off dead black strips at the photo edges before stitching
FIX_VIGNETTING = True     # measure the lens's edge darkening from the overlaps and undo it
MAX_VIGNETTE_BOOST = 1.5  # never brighten any pixel more than this (1.5 = undoes up to 33% darkening)
WARP_SHAPE = "cylindrical"  # what the photos get wrapped onto: "cylindrical" or "spherical"
CROP_BLACK_EDGES = True

# Lens correction. The stitcher assumes straight lines stay straight (a pinhole lens).
# Wide lenses bow them outward, which wrecks the focal length estimate. Both 0 = off.
# Fill these in from a checkerboard calibration (cv2.calibrateCamera) of the real camera.
# Setting LENS_HFOV also switches bundle adjustment to one locked focal length for all photos.
# Navcam test frames: LENS_HFOV = 86, LENS_K1 = -0.08 (best fit against the rover's known
# mast angles, Nov 9; NASA's 96 deg spec is for the raw lens, before this simple correction).
# Overridable with --hfov / --lens-k1.
LENS_HFOV = 0.0           # horizontal field of view in degrees, sets the focal length for the correction
LENS_K1 = 0.0             # barrel distortion; negative straightens outward bowing
MIN_CROP_KEEP = 0.5       # don't crop if it would throw away more than half the image


def link_cutoff():
    """The confidence a photo pair needs to count as linked (looser for in-order shots)."""
    return IN_ORDER_MIN_CONFIDENCE if SEQUENTIAL_MATCHING else MIN_CONFIDENCE


def link_floor():
    """The agreeing-spot count a photo pair needs to count as linked (looser for in-order shots)."""
    return IN_ORDER_MIN_INLIERS if SEQUENTIAL_MATCHING else MIN_INLIERS


def log_time(label):
    """Prints time since launch and how long this stage took."""
    global _last_mark
    now = time.perf_counter()
    print(f"[{now - SCRIPT_START:6.1f} s total | +{now - _last_mark:5.1f} s] {label}")
    _last_mark = now


def load_images(folder):
    """Reads every photo in the folder, sorted by name. Gives back (names, images), or None."""
    if not folder.is_dir():
        print(f"Error: there is no folder at {folder}. Put the photos there or pass --input some_folder.")
        return None

    image_files = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in IMAGE_TYPES and not f.name.startswith(".")  # skip macOS ._ files
    )
    if len(image_files) < 2:
        print(f"Error: need at least two pictures in {folder}, found {len(image_files)}.")
        return None

    names, images = [], []
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Error: could not read {image_file}")
            return None
        names.append(image_file.name)
        images.append(image)
        print(f"Loaded: {image_file.name}")
    return names, images


def undistort_all(images):
    """
    Step 0: straightens the lens bow out of every photo so they act like pinhole shots.
    For barrel distortion (negative k1) every output pixel lands inside the original,
    so no black corners appear; the outermost rim of the photo gets trimmed instead.
    """
    height, width = images[0].shape[:2]
    focal = (width / 2) / math.tan(math.radians(LENS_HFOV / 2))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    bend = np.array([LENS_K1, 0, 0, 0])
    # Work out the pixel shuffle once, then apply it to every photo (same lens, same shuffle)
    map_x, map_y = cv2.initUndistortRectifyMap(K, bend, None, K, (width, height), cv2.CV_32FC1)
    return [cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR) for image in images]


def trim_dark_borders(images):
    """
    Cuts away edge rows and columns that are nearly black in the photos. Some
    sensors read out a strip of dead pixels (Mastcam-Z has ~23 columns on one side),
    and blending those in leaves dark streaks. The same cut goes on every photo so
    they stay the same size.
    """
    cut = [0, 0, 0, 0]  # top, bottom, left, right
    for image in images:
        gray = image.mean(axis=2) if image.ndim == 3 else image.astype(float)
        dark = 0.15 * np.median(gray)  # "nearly black" = under 15% of the photo's typical brightness
        rows, cols = gray.mean(axis=1), gray.mean(axis=0)
        limit_r, limit_c = len(rows) // 10, len(cols) // 10  # never cut more than 10% off a side
        sides = [rows[:limit_r], rows[::-1][:limit_r], cols[:limit_c], cols[::-1][:limit_c]]
        for k, edge in enumerate(sides):
            bright = np.nonzero(edge > dark)[0]
            if len(bright):
                cut[k] = max(cut[k], int(bright[0]) + (2 if bright[0] > 0 else 0))  # 2 px extra past the dim fringe
    if not any(cut):
        return images
    top, bottom, left, right = cut
    print(f"Trimmed dark borders: {top} top, {bottom} bottom, {left} left, {right} right")
    return [image[top:image.shape[0] - bottom, left:image.shape[1] - right] for image in images]


def measure_vignetting(images, features, matches, work_scale):
    """
    Works out how much darker the lens makes the edges than the centre, from the
    photos alone. Where two photos overlap, one spot is seen near the middle of one
    photo and near the edge of the other, so its brightness ratio measures the
    falloff. Fits  log(brightness) = photo gain + a*r^2 + b*r^4,  r = distance from
    the centre (1.0 at the corner). Gives back (a, b).
    """
    small = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=work_scale, fy=work_scale,
                        interpolation=cv2.INTER_AREA).astype(np.float64) for im in images]
    small = [cv2.blur(s, (5, 5)) for s in small]  # average a small patch so noise doesn't dominate

    def sample(k, points):
        """Brightness at each spot in photo k, and its squared distance from centre (1.0 at a corner)."""
        sh, sw = small[k].shape  # photos can differ in size, so measure each one
        x, y = np.clip(points.T.round().astype(int), 0, [[sw - 1], [sh - 1]])
        return small[k][y, x], ((x - sw / 2) ** 2 + (y - sh / 2) ** 2) / ((sw / 2) ** 2 + (sh / 2) ** 2)

    count = len(images)
    rows, targets = [], []
    for i, j, points_i, points_j, _, _ in trusted_point_pairs(features, matches):
        vi, r2i = sample(i, points_i)
        vj, r2j = sample(j, points_j)
        usable = (vi > 15) & (vi < 240) & (vj > 15) & (vj < 240)  # skip clipped black and white
        vi, vj, r2i, r2j = vi[usable], vj[usable], r2i[usable], r2j[usable]
        for a, b, v1, v2 in zip(r2i, r2j, vi, vj):
            row = np.zeros(count + 2)
            row[i], row[j] = 1, -1                    # photo gains
            row[count], row[count + 1] = a - b, a * a - b * b  # falloff terms
            rows.append(row)
            targets.append(math.log(v1) - math.log(v2))
    if len(rows) < 50:
        return 0.0, 0.0
    A, y = np.array(rows), np.array(targets)
    A = np.vstack([A, np.r_[1.0, np.zeros(count + 1)]])  # pin photo 0's gain, or the fit has no anchor
    y = np.r_[y, 0.0]
    weights = np.ones(len(y))
    for _ in range(4):  # refit a few times, each time trusting the worst misfits less
        solution = np.linalg.lstsq(A * weights[:, None], y * weights, rcond=None)[0]
        misfit = np.abs(A @ solution - y)
        scale = 1.5 * np.median(misfit) + 1e-6
        weights = np.where(misfit <= scale, 1.0, scale / np.maximum(misfit, 1e-12))
    return float(solution[count]), float(solution[count + 1])


def fix_vignetting(images, a, b):
    """Brightens each photo's edges by the measured falloff so overlaps match."""
    fixed = []
    for image in images:  # built per photo, since photos can differ in size
        height, width = image.shape[:2]
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        r2 = ((xs - width / 2) ** 2 + (ys - height / 2) ** 2) / ((width / 2) ** 2 + (height / 2) ** 2)
        boost = np.exp(-(a * r2 + b * r2 * r2))  # undo log-falloff a*r^2 + b*r^4
        # Few matched spots sit in the far corners, so the fit is guessing there. Capping
        # stops it blowing out bright things (Navcam measured 62% and burned the white deck).
        boost = np.minimum(boost, MAX_VIGNETTE_BOOST)
        boost = boost[..., None] if image.ndim == 3 else boost
        fixed.append(np.clip(image.astype(np.float32) * boost, 0, 255).astype(np.uint8))
    return fixed


def find_features(images, scale):
    """Step 1: picks out the distinctive spots (corners, specks, texture) in each photo, on shrunk copies."""
    finder = cv2.SIFT_create(
        nfeatures=MAX_FEATURES, contrastThreshold=SIFT_CONTRAST, edgeThreshold=SIFT_EDGE_LIMIT
    )
    features = []
    for index, image in enumerate(images):
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        feature = cv2.detail.computeImageFeatures2(finder, image)
        feature.img_idx = index  # the later steps keep track of photos by this number
        features.append(feature)
    return features


def match_features(features):
    """Step 2: compares photos to see which overlap. Gives back an N x N grid, flattened row by row."""
    matcher = cv2.detail_BestOf2NearestMatcher(try_use_gpu=False, match_conf=MATCH_STRICTNESS)
    count = len(features)
    if SEQUENTIAL_MATCHING:
        index = np.arange(count)
        nearby = np.abs(index[:, None] - index[None, :]) <= NEIGHBOUR_RANGE
        # Also try last-with-first, which closes a full 360 ring. Without it, one weak
        # link or a missing frame splits the ring in two.
        nearby[0, count - 1] = nearby[count - 1, 0] = True
        matches = matcher.apply2(features, cv2.UMat(nearby.astype(np.uint8)))
    else:
        matches = matcher.apply2(features)  # tries every pair, so the file order doesn't matter
    matcher.collectGarbage()
    matches = restore_clean_matches(matches)
    if SEQUENTIAL_MATCHING and count > 2:
        # The closing pair isn't neighbours in a partial pan, so a coincidence could link
        # it. Hold it to the stricter all-pairs bar.
        for match in (matches[count - 1], matches[(count - 1) * count]):
            if match.num_inliers < MIN_INLIERS or match.confidence < MIN_CONFIDENCE:
                match.confidence = 0.0
    return matches


def restore_clean_matches(matches):
    """
    Puts back photo pairs that OpenCV threw out for matching *too* well.

    OpenCV scores a pair as inliers / (8 + 0.3 * matches) and zeroes any score
    over 3, assuming the photos are duplicates. But over 3 just means 90%+ of
    matches agree, which a clean pan from a steady mount gives you. All the Utah
    test frames failed because of this. Cap the score at 3 instead.
    """
    for match in matches:
        if match.src_img_idx >= 0 and match.num_inliers < link_floor():
            match.confidence = 0.0  # too few agreeing spots to trust, whatever the score says
            continue
        if match.src_img_idx < 0 or match.confidence > 0 or match.num_inliers < 50:
            continue  # skip self-pairs, pairs that kept their score, and weak pairs
        score = match.num_inliers / (8 + 0.3 * len(match.matches))
        if score > 3:
            match.confidence = 3.0
    return matches


def drop_stretched_links(matches, count, work_size, names):
    """
    With a known lens, throws out photo links that can't come from a turning camera.
    Undo the lens on a link's homography and what's left should be a pure rotation,
    which doesn't stretch anything. A false link (matched noise) stretches wildly.
    """
    width, height = work_size
    focal = (width / 2) / math.tan(math.radians(LENS_HFOV / 2))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    for i in range(count):
        for j in range(i + 1, count):
            for match in (matches[i * count + j], matches[j * count + i]):  # the grid holds both directions
                if match.confidence <= 0 or match.H is None:
                    continue
                A = K_inv @ np.float64(match.H) @ K
                det = np.linalg.det(A)
                stretch = np.inf
                if abs(det) > 1e-12:
                    s = np.linalg.svd(A / np.cbrt(det), compute_uv=False)
                    stretch = s[0] / max(s[2], 1e-12)  # longest over shortest axis; 1.0 = no stretch
                if stretch > MAX_LINK_STRETCH:
                    if match is matches[i * count + j]:
                        print(f"Dropped link {names[i]}-{names[j]}: it stretches {stretch:.0f}x, "
                              "so it isn't a camera turning (probably noise matched to noise)")
                    match.confidence = 0.0


def warn_blurry(images, names):
    """Names any photo much softer than the rest, so the rover can retake it."""
    sharpness = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        scale = min(1.0, 800 / max(gray.shape))  # judge on a common size, so big photos aren't favoured
        gray = cv2.resize(gray.astype(np.float64), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # At the strongest edges, compare how sharp the edge is (2nd derivative) to how
        # big it is (1st derivative). Blur smears edges, so the ratio drops. A plain wall
        # has weak edges but they're still crisp, so it doesn't trip this (a plain
        # edge-energy score flagged two blank-wall frames in captures/).
        slope = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        strong = slope > np.percentile(slope, 90)
        curvature = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        sharpness.append(curvature[strong].mean() / max(slope[strong].mean(), 1e-9))
    typical = float(np.median(sharpness))
    for name, value in zip(names, sharpness):
        if value < BLUR_WARNING * typical:
            print(f"Warning: {name} looks blurry ({value / typical:.0%} as sharp as the others). "
                  "Retake it with the camera holding still.")


def keep_linked(features, matches, linked):
    """
    Trims the feature and match lists to the photos in `linked`, renumbered 0, 1, 2...

    OpenCV's leaveBiggestComponent does this in C++, but from Python it only hands
    back the indices. Without this, one unmatched photo can wreck the camera
    estimate for every other photo.
    """
    count = len(features)
    renumber = {old: new for new, old in enumerate(linked)}

    for old in linked:
        features[old].img_idx = renumber[old]
    kept_features = [features[old] for old in linked]

    kept_matches = []
    for old_a in linked:
        for old_b in linked:
            match = matches[old_a * count + old_b]
            if match.src_img_idx >= 0:  # -1 is a photo paired with itself
                match.src_img_idx = renumber[match.src_img_idx]
                match.dst_img_idx = renumber[match.dst_img_idx]
            kept_matches.append(match)
    return kept_features, kept_matches


def nearest_rotation(A):
    """Snaps a nearly-rotation matrix (noisy, slightly stretched) to the closest true rotation."""
    U, _, Vt = np.linalg.svd(A)
    if np.linalg.det(U @ Vt) < 0:
        return U @ np.diag([1, 1, -1]) @ Vt  # avoid a mirror image
    return U @ Vt


def trusted_point_pairs(features, matches):
    """For every photo pair we trust, the spots that agreed: [(i, j, points_in_i, points_in_j, H, score)]."""
    count = len(features)
    pairs = []
    for i in range(count):
        for j in range(i + 1, count):
            match = matches[i * count + j]
            if match.confidence < link_cutoff() or match.H is None:
                continue
            agreed = np.asarray(match.inliers_mask).ravel().astype(bool)
            kept = [m for m, ok in zip(match.matches, agreed) if ok]
            if kept:
                points_i = np.float64([features[i].keypoints[m.queryIdx].pt for m in kept])
                points_j = np.float64([features[j].keypoints[m.trainIdx].pt for m in kept])
                pairs.append((i, j, points_i, points_j, np.float64(match.H), match.confidence))
    return pairs


def pixel_rays(points, focal, cx, cy):
    """Turns pixel spots into unit-length pointing directions out of the lens."""
    rays = np.column_stack([(points[:, 0] - cx) / focal, (points[:, 1] - cy) / focal, np.ones(len(points))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def chain_rotations(count, pairs, K):
    """
    First guess at every photo's rotation. Each pair's homography (the 3x3 map from
    one photo onto the other) holds their relative rotation once the focal length
    is known. Start from photo 0 and walk outward along the strongest links.
    """
    K_inv = np.linalg.inv(K)
    rotations = {0: np.eye(3)}
    strongest_first = sorted(pairs, key=lambda pair: -pair[5])
    while len(rotations) < count:
        grew = False
        for i, j, _, _, H, _ in strongest_first:
            relative = nearest_rotation(K_inv @ H @ K)  # = R_j^T R_i
            if i in rotations and j not in rotations:
                rotations[j] = rotations[i] @ relative.T
                grew = True
            elif j in rotations and i not in rotations:
                rotations[i] = rotations[j] @ relative
                grew = True
        if not grew:
            return None
    return [rotations[k] for k in range(count)]


def adjust_rotations(features, matches, focal, cx, cy, iterations=50):
    """
    Bundle adjustment for a camera with a known lens. Every photo shares one locked
    focal length, and only the rotations move. OpenCV's adjuster lets each photo
    pick its own focal length, which on the Navcam rings spread from 354 to 820 px
    and bent the horizon. Gives back (rotations, rms misfit in pixels), or None.
    """
    count = len(features)
    pairs = trusted_point_pairs(features, matches)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    start = chain_rotations(count, pairs, K)
    if start is None:
        return None
    # Each pair's rays only depend on the focal length, which is locked, so work them out once
    pair_rays = [(i, j, pixel_rays(pi, focal, cx, cy), pixel_rays(pj, focal, cx, cy)) for i, j, pi, pj, _, _ in pairs]

    def to_rotations(params):  # photo 0 stays put; the rest are 3 numbers each (axis x angle)
        return [np.eye(3)] + [cv2.Rodrigues(params[3 * k:3 * k + 3])[0] for k in range(count - 1)]

    def misfit(params):
        rotations = to_rotations(params)
        # A matched spot seen from photo i and from photo j should point the same way in the world
        gaps = np.concatenate([
            ((rays_i @ rotations[i].T) - (rays_j @ rotations[j].T)).ravel() * focal  # ~pixels
            for i, j, rays_i, rays_j in pair_rays
        ])
        # Huber weighting: big misfits (parallax from the rover's own deck) get a softer vote
        size = np.abs(gaps)
        return gaps * np.where(size <= 2.0, 1.0, np.sqrt(2.0 / np.maximum(size, 1e-9)))

    # Levenberg-Marquardt: Newton steps when the fit is going well, cautious small steps when not
    params = np.concatenate([cv2.Rodrigues(R)[0].ravel() for R in start[1:]])
    gaps = misfit(params)
    cost, damping = gaps @ gaps, 1e-3
    for _ in range(iterations):
        jacobian = np.empty((len(gaps), len(params)))
        for k in range(len(params)):  # only 3 numbers per photo, so a finite difference is cheap
            nudge = np.zeros_like(params)
            nudge[k] = 1e-6
            jacobian[:, k] = (misfit(params + nudge) - gaps) / 1e-6
        normal, gradient = jacobian.T @ jacobian, jacobian.T @ gaps
        gain = 0.0
        while damping < 1e8:
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), -gradient)
            new_gaps = misfit(params + step)
            new_cost = new_gaps @ new_gaps
            if new_cost < cost:
                gain = cost - new_cost
                params, gaps, cost, damping = params + step, new_gaps, new_cost, damping * 0.3
                break
            damping *= 10
        if gain < 1e-6 * cost:
            break
    return to_rotations(params), math.sqrt(cost / len(gaps))


def find_cameras(features, matches, work_size):
    """Step 3: figures out which way the camera pointed for each photo. Gives back None if it can't."""
    estimator = cv2.detail_HomographyBasedEstimator()
    ok, cameras = estimator.apply(features, matches, None)
    if not ok:
        print("Stitching failed: could not estimate the camera directions.")
        return None
    for camera in cameras:
        camera.R = camera.R.astype(np.float32)  # the next step crashes unless the rotations are 32-bit

    if LENS_HFOV > 0:
        # Known lens: lock the focal length and solve only the rotations
        width, height = work_size
        focal = (width / 2) / math.tan(math.radians(LENS_HFOV / 2))
        solved = adjust_rotations(features, matches, focal, width / 2, height / 2)
        if solved is None:
            print("Stitching failed: the photo links don't connect into one chain.")
            return None
        rotations, rms = solved
        print(f"Rotations fitted with the focal length locked, misfit {rms:.1f} px")
        for camera, rotation in zip(cameras, rotations):
            camera.R = rotation.astype(np.float32)
            camera.focal, camera.aspect = focal, 1.0
            camera.ppx, camera.ppy = width / 2, height / 2
    else:
        # Unknown lens: OpenCV's bundle adjustment, which also guesses a focal length per photo
        adjuster = cv2.detail_BundleAdjusterRay()
        adjuster.setConfThresh(link_cutoff())
        ok, cameras = adjuster.apply(features, matches, cameras)
        if not ok:
            print("Stitching failed: bundle adjustment could not line the cameras up.")
            return None

    # Wave correction: levels out the horizon so the panorama doesn't come out curved
    rotations = cv2.detail.waveCorrect(
        [np.copy(camera.R) for camera in cameras], cv2.detail.WAVE_CORRECT_HORIZ
    )
    for camera, rotation in zip(cameras, rotations):
        camera.R = rotation
    return cameras


def pixel_azimuths(image, camera, step=16):
    """Which way (degrees left/right, -180 to 180) each pixel looks, on a coarse grid. Columns x rows."""
    height, width = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(0, width, step, dtype=np.float64), np.arange(0, height, step, dtype=np.float64))
    K_inv = np.linalg.inv(camera_matrix(camera).astype(np.float64))
    rays = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ (np.float64(camera.R) @ K_inv).T
    return np.degrees(np.arctan2(rays[..., 0], rays[..., 2])), xs[0]


def handle_full_ring(images, cameras):
    """
    A full 360 ring has no natural ends, but the cylinder has a back seam where its
    left and right edges meet (straight behind, 180 deg). Any photo that crosses it
    gets wrapped onto both edges at once, which made a 4300 px-wide footprint and a
    black hole in the blend. So: turn the whole ring until the seam sits in the
    widest gap between neighbouring photos, then trim off the sliver of any photo
    still crossing it. Its neighbour covers that sliver anyway.
    """
    centres = sorted(
        (math.degrees(math.atan2(*(np.float64(c.R) @ [0, 0, 1.0])[[0, 2]])), k) for k, c in enumerate(cameras)
    )
    angles = [a for a, _ in centres]
    gaps = [((angles[(k + 1) % len(angles)] - angles[k]) % 360, k) for k in range(len(angles))]
    widest, k = max(gaps)
    if widest > 120 or len(cameras) < 3:
        return images  # a big empty stretch means it's a partial pan, not a ring
    seam_now = angles[k] + widest / 2  # middle of the widest gap
    turn = math.radians(180 - seam_now)
    spin = np.array([[math.cos(turn), 0, math.sin(turn)], [0, 1, 0], [-math.sin(turn), 0, math.cos(turn)]])
    for camera in cameras:
        camera.R = (spin @ np.float64(camera.R)).astype(np.float32)

    trimmed = []
    for image, camera in zip(images, cameras):
        az, columns = pixel_azimuths(image, camera)
        if az.max() - az.min() < 180:  # doesn't cross the seam
            trimmed.append(image)
            continue
        # Each column is on the right side (positive) or left side (negative) of the seam
        side = np.sign(az)
        keep_right = (side > 0).sum() >= (side < 0).sum()
        clean = np.all(side == (1 if keep_right else -1), axis=0)  # columns wholly on the kept side
        kept = columns[clean]
        if len(kept) == 0:
            trimmed.append(image)
            continue
        x0, x1 = int(kept.min()), min(int(kept.max()) + 16, image.shape[1])
        camera.ppx -= x0  # cutting off the left part moves the centre left in the new picture
        trimmed.append(image[:, x0:x1])
    return trimmed


def cameras_look_sane(cameras, images):
    """
    Sanity check before the heavy lifting. A bad alignment can give a focal length
    way off, which asks for a canvas of hundreds of megapixels and runs the
    computer out of memory.
    """
    focals = [camera.focal for camera in cameras]
    typical = float(np.median(focals))
    if min(focals) < 0.6 * typical or max(focals) > 1.6 * typical:  # one camera, so these should agree
        print("Stitching failed: the photos disagree on the focal length.")
        return False

    corners, sizes = footprints(images, cameras)
    _, _, width, height = cv2.detail.resultRoi(corners, sizes)
    canvas_area = width * height
    photo_area = sum(image.shape[0] * image.shape[1] for image in images)
    if canvas_area > MAX_CANVAS_RATIO * photo_area or canvas_area > MAX_CANVAS_MEGAPIX * 1e6:
        print(f"Stitching failed: the alignment asks for a {width} x {height} canvas, too big to be right.")
        return False
    return True


def make_warper(cameras, shrink=1.0):
    """The cylinder all the photos get wrapped onto, sized off the typical focal length."""
    focal = float(np.median([camera.focal for camera in cameras]))
    return cv2.PyRotationWarper(WARP_SHAPE, focal * shrink)


def camera_matrix(camera, shrink=1.0):
    K = camera.K().astype(np.float32)
    K[:2, :] *= shrink  # shrinking a photo is the same as shortening the focal length
    return K


def footprints(images, cameras):
    """Where each photo will land on the full-size cylinder, worked out without bending any pixels."""
    warper = make_warper(cameras)
    boxes = [
        warper.warpRoi((image.shape[1], image.shape[0]), camera_matrix(camera), camera.R)
        for image, camera in zip(images, cameras)
    ]
    corners = [(x, y) for x, y, _, _ in boxes]
    sizes = [(w, h) for _, _, w, h in boxes]
    return corners, sizes


def warp_one(warper, image, camera, shrink=1.0):
    """Wraps one photo onto the cylinder. Gives back its top-left corner, the bent photo, and its pixel mask."""
    K = camera_matrix(camera, shrink)
    if shrink != 1.0:
        image = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    corner, warped = warper.warp(image, K, camera.R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
    blank = np.full(image.shape[:2], 255, np.uint8)
    _, mask = warper.warp(blank, K, camera.R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
    return corner, warped, mask


def plan_seams(images, cameras):
    """
    Step 4a: decides how to even out the brightness and where each seam goes.
    Runs on small copies on purpose: the seam cut at full size eats several GB
    of memory, and the seams barely move.
    """
    height, width = images[0].shape[:2]
    shrink = min(1.0, (SEAM_PIXELS / (height * width)) ** 0.5)
    warper = make_warper(cameras, shrink)
    corners, warped, masks = zip(*(
        warp_one(warper, image, camera, shrink) for image, camera in zip(images, cameras)
    ))
    corners = list(corners)
    masks = [cv2.UMat(m) for m in masks]

    compensator = cv2.detail.ExposureCompensator_createDefault(
        cv2.detail.ExposureCompensator_GAIN_BLOCKS
    )
    compensator.feed(corners, [cv2.UMat(w) for w in warped], masks)

    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    seams = seam_finder.find([cv2.UMat(w.astype(np.float32)) for w in warped], corners, masks)
    return compensator, [seam.get() for seam in seams]


def blend_bands(corners, sizes):
    """How many blend layers to use, scaled to the panorama size (OpenCV Stitcher's rule)."""
    _, _, width, height = cv2.detail.resultRoi(corners, sizes)
    blend_width = math.sqrt(width * height) * BLEND_STRENGTH / 100
    return max(1, math.ceil(math.log2(max(blend_width, 1))) - 1)


def blend(images, cameras, compensator, seams):
    """Step 4b: wraps each photo at full size, fixes its brightness and seam, and blends them."""
    corners, sizes = footprints(images, cameras)
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(blend_bands(corners, sizes))
    blender.prepare(cv2.detail.resultRoi(corners, sizes))

    warper = make_warper(cameras)
    for i, (image, camera) in enumerate(zip(images, cameras)):  # one full-size photo in memory at a time
        corner, warped, mask = warp_one(warper, image, camera)
        warped = compensator.apply(i, corner, warped, mask)
        seam = cv2.dilate(seams[i], None)  # grow a pixel so scaling up leaves no gaps between seams
        seam = cv2.resize(seam, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)
        blender.feed(warped.astype(np.int16), cv2.bitwise_and(seam, mask), corner)  # the blender only takes 16-bit

    panorama, filled = blender.blend(None, None)
    # The blend overshoots 0-255 a little at sharp edges, so clamp it.
    # Absolute value would turn a -4 into a bright speck.
    return np.clip(panorama, 0, 255).astype(np.uint8), filled


def crop_black_edges(panorama, filled):
    """Step 4c: trims one edge at a time, always the blackest, until no black border is left."""
    filled = filled > 0
    top, bottom, left, right = 0, filled.shape[0], 0, filled.shape[1]
    while top < bottom and left < right:
        region = filled[top:bottom, left:right]
        gaps = {
            "top": (~region[0, :]).mean(),
            "bottom": (~region[-1, :]).mean(),
            "left": (~region[:, 0]).mean(),
            "right": (~region[:, -1]).mean(),
        }
        worst = max(gaps, key=gaps.get)
        if gaps[worst] == 0:
            break
        if worst == "top":
            top += 1
        elif worst == "bottom":
            bottom -= 1
        elif worst == "left":
            left += 1
        else:
            right -= 1

    cropped = panorama[top:bottom, left:right]
    if cropped.size < MIN_CROP_KEEP * panorama.size:
        print(
            f"Warning: cropping would keep only {cropped.size / panorama.size:.0%} of the image, "
            "so the black edges were left in. The camera was probably tilted."
        )
        return panorama
    return cropped


def create_panorama(images, names=None):
    """Runs every step in order. Gives back the finished panorama, or None if stitching failed."""
    cv2.setRNGSeed(0)  # same photos give the same panorama every run
    names = names or [f"photo {i + 1}" for i in range(len(images))]

    if ROTATE_PHOTOS:
        # The stitcher can't tell which way is up, so an upside-down mount gives an
        # upside-down panorama. The mount is fixed, so turn the photos upright first.
        turn = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        images = [cv2.rotate(image, turn[ROTATE_PHOTOS]) for image in images]
    if TRIM_DARK_BORDERS:
        images = trim_dark_borders(images)
    if LENS_K1 != 0 and LENS_HFOV > 0:
        images = undistort_all(images)
        log_time(f"Lens corrected (k1 = {LENS_K1}, {LENS_HFOV} deg field of view)")

    height, width = images[0].shape[:2]
    work_scale = min(1.0, (WORK_PIXELS / (height * width)) ** 0.5)
    print(f"Matching on copies scaled to {work_scale:.2f}x")
    warn_blurry(images, names)
    features = find_features(images, work_scale)
    log_time("Features found")
    matches = match_features(features)
    work_size = (round(width * work_scale), round(height * work_scale))  # size of the shrunk copies
    if LENS_HFOV > 0:
        drop_stretched_links(matches, len(images), work_size, names)

    # Keep the biggest group of photos that all link together and drop any stragglers
    linked = np.array(cv2.detail.leaveBiggestComponent(features, matches, link_cutoff())).ravel().tolist()
    if len(linked) < 2:
        print("Stitching failed: no photos overlap enough.")
        return None
    if len(linked) < len(images):
        dropped = [name for i, name in enumerate(names) if i not in linked]
        print(f"Warning: only {len(linked)} of {len(images)} photos linked up. Dropped: {', '.join(dropped)}")
        features, matches = keep_linked(features, matches, linked)
    images = [images[i] for i in linked]
    log_time("Matching done")

    if FIX_VIGNETTING:
        a, b = measure_vignetting(images, features, matches, work_scale)
        corner = 1 - math.exp(a + b)  # how much darker the corners are than the centre
        if corner > 0.03:  # only bother when it's visible
            images = fix_vignetting(images, a, b)
            log_time(f"Vignetting fixed (corners were {corner:.0%} darker)")

    cameras = find_cameras(features, matches, work_size)
    if cameras is None:
        return None
    for camera in cameras:  # solved on shrunk copies, so scale back up to full-size pixels
        camera.focal /= work_scale
        camera.ppx /= work_scale
        camera.ppy /= work_scale
    images = handle_full_ring(images, cameras)
    if not cameras_look_sane(cameras, images):
        return None
    log_time("Cameras aligned")

    compensator, seams = plan_seams(images, cameras)
    log_time("Seams planned")
    panorama, filled = blend(images, cameras, compensator, seams)
    log_time("Blending done")
    return crop_black_edges(panorama, filled) if CROP_BLACK_EDGES else panorama


def parse_args():
    parser = argparse.ArgumentParser(description="Stitch overlapping photos into one panorama.")
    parser.add_argument("--input", type=Path, default=IMAGE_FOLDER,
                        help="folder of photos (default: captures/ next to this script)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="where to save the panorama (default: panorama.jpg next to this script)")
    parser.add_argument("--hfov", type=float, default=None,
                        help="camera's horizontal field of view in degrees, for lens correction")
    parser.add_argument("--lens-k1", type=float, default=None,
                        help="lens barrel distortion (negative straightens bowed lines), 0 = off")
    parser.add_argument("--warp", choices=["cylindrical", "spherical"], default=None,
                        help="shape the photos get wrapped onto (default: WARP_SHAPE)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None,
                        help="turn every photo clockwise first, to match how the camera is mounted")
    parser.add_argument("--in-order", action="store_true",
                        help="photos were taken in file-name order (servo pan): only compare neighbours")
    parser.add_argument("--show", action="store_true",
                        help="open a window with the result (needs a screen, so not on the rover)")
    return parser.parse_args()


def main():
    """Returns 0 on success and 1 on failure, so other scripts and the rover can tell."""
    global LENS_HFOV, LENS_K1, WARP_SHAPE, SEQUENTIAL_MATCHING, ROTATE_PHOTOS
    args = parse_args()
    if args.rotate is not None:
        ROTATE_PHOTOS = args.rotate
    if args.in_order:
        SEQUENTIAL_MATCHING = True
    if args.hfov is not None:
        LENS_HFOV = args.hfov
    if args.lens_k1 is not None:
        LENS_K1 = args.lens_k1
    if args.warp is not None:
        WARP_SHAPE = args.warp
    print("PROGRAM STARTED")

    print("Loading images...")
    loaded = load_images(args.input)
    if loaded is None:
        return 1
    names, images = loaded
    log_time("Images loaded")

    print("\nCreating panorama...")
    panorama = create_panorama(images, names)
    if panorama is None:
        print(f"\nFAILED after {time.perf_counter() - SCRIPT_START:.1f} s")
        return 1

    try:
        saved = cv2.imwrite(str(args.output), panorama)
    except cv2.error:  # an unknown file extension throws instead of returning False
        saved = False
    if not saved:
        print(f"Error: could not save the panorama to {args.output}")
        return 1
    log_time("Panorama saved")  # clock stops here, before the --show window waits on a key
    print("\nPanorama created successfully!")
    print(f"Saved as: {args.output}")

    if args.show:
        cv2.namedWindow("Science Module Panorama", cv2.WINDOW_NORMAL)  # resizable, so a wide image fits on screen
        cv2.imshow("Science Module Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
