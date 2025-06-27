from tqdm import tqdm
import numpy as np
from pathlib import Path
from skimage.measure import regionprops
import geomdl
from geomdl import NURBS
from skimage.morphology import binary_erosion, disk, binary_dilation
from skimage.measure import find_contours
from skimage.measure import label as ski_label
import random
from shapely.geometry import Polygon, LineString
from skimage.filters import gaussian
import yaml
import copy
from collections import defaultdict
from tqdmp import tqdmp
import warnings
import pandas as pd
import argparse
from medvol import MedVol


def generate_scribble_dataset(load_dir, save_dir, num_labels, conf_filepath, num_processes, disable_ignore, names=None):
    load_dir = Path(load_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    num_labels = int(num_labels)

    with open(conf_filepath, "r") as yaml_file:
        conf = yaml.safe_load(yaml_file)

    if num_processes is not None:
        num_processes = int(num_processes)    

    if names is None:
        names = [path.name[:-7] for path in load_dir.rglob("*.nii.gz")]
    elif isinstance(names, str):
        names = [names]

    for class_type in ["foreground", "background"]:
        class_probabilities = conf[class_type]["class_probabilities"]
        num_class_type_labels = num_labels-1 if class_type == "foreground" else 1
        if not isinstance(class_probabilities, list):
            if class_type == "foreground":
                conf[class_type]["class_probabilities"] = {i+1: 1 for i in range(num_class_type_labels)}
            else:
                conf[class_type]["class_probabilities"] = {0: 1}
        if not isinstance(conf[class_type]["num_slice_range"], list):
            conf[class_type]["num_slice_range"] = [conf[class_type]["num_slice_range"], conf[class_type]["num_slice_range"]]

    tqdmp(generate_scribble_seg, names, num_processes, load_dir=load_dir, save_dir=save_dir, conf=conf, num_labels=num_labels, disable_ignore=disable_ignore)


def generate_scribble_seg(name, load_dir, save_dir, conf, num_labels, disable_ignore):
    seg = MedVol(load_dir / f"{name}.nii.gz")
    scribbles_array = seg2scribbles(copy.deepcopy(seg.array), conf, name, num_labels, disable_ignore)
    scribbles = MedVol(scribbles_array, copy=seg)
    scribbles.save(save_dir / f"{name}.nii.gz")


def seg2scribbles(seg, conf, name, num_labels, disable_ignore, verbose=False):

    scribble_thickness = conf["generation_properties"]["scribble_thickness"]
    if scribble_thickness <= 0 or scribble_thickness % 2 == 0:
        raise RuntimeError("Scribble thickness needs to be an odd number and larger than zero.")
    scribble_thickness = int((scribble_thickness - 1) / 2)
    
    # Components: Label, slice, component
    components = compute_components(seg, list(range(num_labels)), conf["filter"]["min_area"])
    components = choose_components(seg, components, copy.deepcopy(conf))
    components = flatten_components(components)
    components = dicts2dataframe(components, name)
    scribbles = np.zeros_like(seg)
    if not disable_ignore:
        scribbles += num_labels

    for index, component in tqdm(components.iterrows(), total=len(components), disable=not verbose):
        xground = "background" if component["label"] == 0 else "foreground"
        for mode in conf[xground]["scribbles"].keys():
            num_scribbles = conf[xground]["scribbles"][mode]["num_scribbles"]
            scribble_generator = mode2scribble_generator(mode)
            for _ in range(num_scribbles):
                scribble = scribble_generator(component["mask"], component["prop"], scribble_thickness, **conf[xground]["scribbles"][mode]["params"])
                if scribble is not None:
                    scribble = binary_dilation(scribble, disk(scribble_thickness))
                    scribble[component["mask"] == 0] = 0
                    if not disable_ignore:
                        scribbles[component["slice"]][scribble == 1] = component["label"]
                    else:
                        scribbles[component["slice"]][scribble == 1] = num_labels if component["label"] == 0 else component["label"]

    return scribbles


def compute_components(seg, labels, min_area):
    # Components: Label, slice, component
    slices = {}
    for slice_index in range(len(seg)):
        current_slice = seg[slice_index]
        components = separate_components(current_slice)
        components = filter_components(components, min_area)
        slices[slice_index] = components
    label_slice_component_dict = defaultdict(dict)
    for label in labels:
        for slice_index, slice_components in slices.items():
            if label in slice_components:
                c = [{"prop": slice_components[label]["props"][i], "mask": slice_components[label]["masks"][i]} for i in range(len(slice_components[label]["props"]))]
                label_slice_component_dict[label][slice_index] = c
    return label_slice_component_dict


def separate_components(seg):
    # Components: Label, slice, component
    labels = np.unique(seg)
    components = {}
    for label in labels:
        label_components = ski_label(seg == label)
        props = regionprops(label_components)
        masks = [label_components == prop["label"] for prop in props]
        components[label] = {"props": props, "masks": masks}
    return components


def filter_components(components, min_area):
    # Components: Label, slice, component
    filtered_components = {}
    for label in components.keys():
        filtered_props, filtered_masks = [], []
        for prop, mask in zip(components[label]["props"], components[label]["masks"]):
            if prop["area"] >= min_area:
                filtered_props.append(prop)
                filtered_masks.append(mask)
        if len(filtered_props) > 0:
            filtered_components[label] = {"props": filtered_props, "masks": filtered_masks}
    return filtered_components


def choose_components(seg, components, conf):
    # Components: Label, slice, component
    components = dict(components)
    num_foreground_slices = np.sum([np.any(seg_slice) for seg_slice in seg])
    num_background_slices = len(seg) - num_foreground_slices
    # print("Num foreground slices (components): ", len(np.unique(np.concatenate([list(components[1].keys()), list(components[2].keys()), list(components[3].keys())]))))
    # print("Num foreground slices (seg): ", num_foreground_slices)
    background_slices = [i for i in range(len(seg)) if not np.any(seg[i])]
    chosen_components = defaultdict(lambda: defaultdict(list))
    num_foreground_slices = int(num_foreground_slices * random.uniform(*conf["foreground"]["num_slice_range"]))
    num_background_slices = int(num_background_slices * random.uniform(*conf["background"]["num_slice_range"]))
    slices = ["foreground"] * num_foreground_slices + ["background"] * num_background_slices
    conf["foreground"]["class_probabilities"] = {label: prob for label, prob in conf["foreground"]["class_probabilities"].items() if label in components.keys()}
    labels = np.unique(seg)

    for i, slice_type in enumerate(slices):
        chosen_label = random.choices(list(conf[slice_type]["class_probabilities"].keys()), list(conf[slice_type]["class_probabilities"].values()))[0]
        if slice_type == "foreground":
            chosen_slice = random.choice(list(components[chosen_label].keys()))
        else:
            slice_candidates = [curr_slice for curr_slice in components[chosen_label].keys() if curr_slice in background_slices]
            chosen_slice = random.choice(slice_candidates)

        for label in labels:
            if (label in components) and (chosen_slice in components[label]):
                component_selection_probs = np.array([np.log(component["prop"].area) for component in components[label][chosen_slice]])
                component_selection_probs = np.exp(component_selection_probs) / sum(np.exp(component_selection_probs))  # Convert to probabilities with softmax
                chosen_component_index = np.random.choice(range(len(components[label][chosen_slice])), p=component_selection_probs)
                chosen_components[label][chosen_slice].append(components[label][chosen_slice][chosen_component_index])

                del components[label][chosen_slice]
                if len(components[label].keys()) == 0:
                    del components[label]
                    if label in conf[slice_type]["class_probabilities"]:
                        del conf[slice_type]["class_probabilities"][label]
    
    return chosen_components


def flatten_components(components):
    flattended_components = []

    for label in components.keys():
        for slice_index in components[label].keys():
            for component in components[label][slice_index]:
                flattended_components.append({"label": label, "slice": slice_index, "prop": component["prop"], "mask": component["mask"]})

    return flattended_components


def dicts2dataframe(components, name):
    components = pd.DataFrame.from_dict({
        'name': [name] * len(components), 
        'label': [component['label'] for component in components],
        'slice': [component['slice'] for component in components],
        'prop': [component['prop'] for component in components],
        'mask': [component['mask'] for component in components],
        })
    return components


def gen_nurbs_scribble(mask, prop, contour_distance, num_control_points, curve_delta):
    # Get coordiantes
    coords = prop.coords
    # Randomly select N points from the region
    # points = coords[np.random.choice(coords.shape[0], num_control_points, replace=False)]
    points = generate_control_points(mask, coords, num_control_points)
    # Create a NURBS curve instance
    curve = NURBS.Curve()
    # Set degree
    curve.degree = 2
    # Set control points
    curve.ctrlptsw = [list(pt) + [1.0] for pt in points]  # Weights are all set to 1.0
    # Auto-generate knot vector
    curve.knotvector = geomdl.utilities.generate_knot_vector(curve.degree, len(curve.ctrlptsw))
    # Set delta (this affects the number of points in the evaluated curve)
    curve.delta = curve_delta
    # Evaluate curve
    curve = np.array(curve.evalpts)
    scribble = render_curve(curve, mask.shape)
    scribble[mask == 0] = 0
    return scribble


def gen_contour_scribble(mask, prop, contour_distance, disk_size_range, scribble_length_range):
    # Erode mask
    disk_size_range = random.randint(contour_distance+disk_size_range[0], contour_distance+disk_size_range[1])
    # disk_size_range = contour_distance+disk_size_range[0]
    eroded_mask = binary_erosion(mask, disk(disk_size_range))
    if not np.any(np.nonzero(eroded_mask)):
        return None
    # Compute curvature of the contour
    contour = find_contours(eroded_mask)
    # contour, _ = cv2.findContours(eroded_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour = [np.flip(c).squeeze(1) for c in contour]
    if len(contour) == 0:
        return None
    contour = np.concatenate(contour, axis=0)
    # Compute curvature of the contour
    curvature = compute_curvature(contour)
    # Compute scribble length
    min_length = int(len(contour)*scribble_length_range[0])
    max_length = int(len(contour)*scribble_length_range[1])
    min_length = min_length if min_length > 0 else 1
    max_length = max_length if max_length > 0 else 1
    length = random.randint(min_length, max_length)
    # Choose scribble position on contour
    scribble_pos = random.choices(range(len(curvature)), curvature)[0]
    scribble_selection = (scribble_pos-int(length/2), scribble_pos+length-int(length/2))
    # Extract scribble
    contour = np.take(contour, range(*scribble_selection), axis=0, mode='wrap')
    contour = np.round(contour).astype(np.int32)
    # Render scribble
    scribble = np.zeros_like(mask)
    scribble[contour[:, 0], contour[:, 1]] = 1
    # It is not guaranteed that the scribble is not a set of scribbles, so we remove all but the largest one
    scribble_components = ski_label(scribble)
    labels, counts = np.unique(scribble_components, return_counts=True)
    counts = counts[labels > 0]
    labels = labels[labels > 0]
    label = labels[np.argmax(counts)]
    scribble = scribble_components == label
    return scribble


def mode2scribble_generator(mode):
    if mode == 'nurbs':
        scribble_generator = gen_nurbs_scribble
    elif mode == 'contour':
        scribble_generator = gen_contour_scribble
    else:
        raise RuntimeError("Unknown scribble generator.")
    return scribble_generator


def generate_control_points(mask, coords, num_points):
    choose_random = False
    if mask.all():
        choose_random = True
        hull = None
    else:
        contours = find_contours(mask)[0]
        hull = Polygon(contours)

    if hull is not None and not hull.is_valid:
        choose_random = True
    
    if choose_random:
        # print("Could not generate visible hull for control points. Falling back to random control points.")
        points = coords[np.random.choice(coords.shape[0], num_points, replace=False)]
        return points

    control_points = []
    while len(control_points) < num_points:
        random_point = coords[np.random.choice(coords.shape[0], 1, replace=False)][0]
        if len(control_points) > 0:
            line = LineString([control_points[-1], random_point])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                intersection = line.intersection(hull)
            # intersection = line.intersection(hull)
            if intersection.geom_type == 'Point':
                intersection_point = intersection
            elif intersection.geom_type == 'MultiPoint':
                # intersection_point = intersection[0]
                continue
            else:
                intersection_point = None

            if intersection_point is None:
                control_point = random_point
            else:
                intersection_line = LineString([control_points[-1], intersection_point])
                random_distance = random.uniform(0, intersection_line.length)
                control_point = line.interpolate(random_distance)
                control_point = (control_point.x, control_point.y)

            control_points.append(control_point)
        else:
            control_points.append(random_point)
    return control_points


def compute_curvature(contour):
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = (dx * d2y - dy * d2x) / np.power(dx**2 + dy**2, 1.5)
    curvature = np.abs(curvature)
    curvature = gaussian(curvature)
    curvature = normalize(curvature)
    return curvature


def render_curve(curve, seg_shape):
    curve = np.round(curve).astype(int)
    scribble = np.zeros(seg_shape, dtype=np.uint8)
    for i in range(1, len(curve)):
        x0, y0 = curve[i-1]
        x1, y1 = curve[i]
        line_pixels = bresenham_line(x0, y0, x1, y1)
        for px, py in line_pixels:
            if 0 <= px < scribble.shape[0] and 0 <= py < scribble.shape[1]:  # Check for valid indices
                scribble[px, py] = 1
    return scribble


# Bresenham's line algorithm
def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        
    points.append((x, y))
    return points


def normalize(x, source_limits=None, target_limits=None):
    if source_limits is None:
        source_limits = (x.min(), x.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return x * 0
    else:
        x_std = (x - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, help="Path to the input segmentation folder.")
    parser.add_argument('-o', "--output", required=True, help="Path to the output scribble folder.")
    parser.add_argument('-l', "--num_labels", required=True, type=int, help="The number of segmentation labels.")
    parser.add_argument('-c', "--conf", required=False, default="scribble_conf.yml", help="(Optional) Path to the scribble configuration.")
    parser.add_argument('-n', "--name", required=False, type=str, default=None, nargs="+", help="List of segmentation names separated by spaces without file extension. All segmentations are used if 'None'.")
    parser.add_argument('-p', "--processes", required=False, type=int, default=None, help="(Optional) Number of multiprocessing processes.")
    parser.add_argument('--disable_ignore', required=False, default=False, action="store_true", help="(Optional) Whether to define an ignore label or not.")
    args = parser.parse_args()

    generate_scribble_dataset(args.input, args.output, args.num_labels, args.conf, args.processes, args.disable_ignore, args.name)