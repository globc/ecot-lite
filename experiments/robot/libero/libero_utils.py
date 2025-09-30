"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import gym
import imageio
import numpy as np
import tensorflow as tf
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark.mu_creation import *  # noqa
from libero.libero.envs.bddl_base_domain import BDDLUtils
from libero.libero.envs.objects import get_object_dict
from libero.libero.utils import task_generation_utils
from libero.libero.utils.bddl_generation_utils import (
    get_object_dict as get_object_num_dict,
)
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import get_scene_class, get_scene_dict, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

def generate_mu_with_distractor_objects(mu_cls, min_distractors, max_distractors, distractor_seed):
    """Generate a version of the initial state distribution with distractor objects. The number and position of the
    distractor objects are determined by distractor_seed.
    Notes:
    1) In the context of an environment generated with mu_cls returned by this function, env.reset() will
    NOT randomize distractors. Instead, the distractors will be fixed for the duration of the environment's lifetime.
    If randomizing the distractors is desired, a new environment should be generated with a different distractor_seed.
    2) If one wishes to prevent correlations between the distractors chosen for different environments, different
    distractor_seed values should be used for the different environments.
    """

    assert max_distractors >= min_distractors

    large_objects = {
        "basket",
        "basin_faucet",
        "chefmate_8_frypan",
        "desk_caddy",
        "dining_set_group",
        "faucet",
        "flat_stove",
        "microwave",
        "rack",
        "short_cabinet",
        "short_fridge",
        "slide_cabinet",
        "white_cabinet",
        "white_storage_box",
        "window",
        "wine_rack",
        "wooden_cabinet",
        "wooden_shelf",
        "wooden_tray",
        "wooden_two_layer_shelf",
    }
    exclude_objects = large_objects.union({"cherries", "corn", "mayo", "salad_dressing", "target_zone"})

    rng = np.random.default_rng(distractor_seed)
    all_object_categories = set(get_object_dict().keys())
    mu = mu_cls()
    object_categories_already_used = []
    for category_name in mu.fixture_object_dict.keys():
        object_categories_already_used.append(category_name)
    for category_name in mu.movable_object_dict.keys():
        object_categories_already_used.append(category_name)
    object_categories_already_used = set(object_categories_already_used)

    remaining_objects = sorted(list(all_object_categories - object_categories_already_used - exclude_objects))

    # Choose a subset of objects to be distractors
    num_distractors = rng.integers(min_distractors, max_distractors + 1)
    selected_objects = list(rng.choice(remaining_objects, num_distractors, replace=False))

    print(
        f"Selected {num_distractors} distractor(s) from {len(remaining_objects)} remaining objects: {selected_objects}"
    )

    class MuWithDistractorObjects(mu_cls):
        def __init__(self, *args, **kwargs):

            # Temporarily set this to False so superclass can initialize properly
            self._include_distractor_in_init_states = False
            super().__init__(*args, **kwargs)
            self._include_distractor_in_init_states = True

            self.distractor_objects = selected_objects
            distractor_object_num_info = {category: 1 for category in self.distractor_objects}

            self.movable_object_dict.update(get_object_num_dict(distractor_object_num_info))
            self.define_regions()

        @property
        def init_states(self):
            if self._include_distractor_in_init_states:
                states = super().init_states
                for distractor_object_category in self.distractor_objects:
                    assert len(self.movable_object_dict[distractor_object_category]) == 1
                    object_name = self.movable_object_dict[distractor_object_category][0]
                    states.append(("On", object_name, f"{self.workspace_name}_{distractor_object_category}_init_region"))
                return states
            else:
                return super().init_states

        def define_regions(self):
            def is_point_in_ranges(x, y, ranges):
                for x1, y1, x2, y2 in ranges:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        return True
                return False

            def sample_point_outside_ranges(ranges, xlim, ylim):
                i = 0
                while i < 5000:
                    x = rng.uniform(*xlim)
                    y = rng.uniform(*ylim)
                    if not is_point_in_ranges(x, y, ranges):
                        return x, y
                    i += 1
                raise ValueError("Could not sample point outside of ranges")

            if self._include_distractor_in_init_states:
                super().define_regions()
                for distractor_object_category in self.distractor_objects:
                    assert len(self.movable_object_dict[distractor_object_category]) == 1

                    # Pick and x, y outside of the current regions
                    current_ranges = []
                    for region in self.xy_region_kwargs_list:
                        if "init_region" not in region["region_name"]:
                            continue
                        radius = 0.15
                        if any(object_category in region["region_name"] for object_category in large_objects):
                            radius = 0.25
                        x1, y1, x2, y2 = region["ranges"][0]
                        current_ranges.append((x1 - radius, y1 - radius, x2 + radius, y2 + radius))
                    x, y = sample_point_outside_ranges(current_ranges, xlim=(-0.3, 0.3), ylim=(-0.275, 0.275))
                    self.regions.update(
                        self.get_region_dict(
                            region_centroid_xy=[x, y],
                            region_name=f"{distractor_object_category}_init_region",
                            target_name=self.workspace_name,
                            region_half_len=0.00001,
                        )
                    )
                    self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)
                    
            else:
                super().define_regions()

    return MuWithDistractorObjects


def generate_expanded_mu(mu_cls, expansion_obj_of_interest, expansion_half_len_factor):
    """Generate a version of the initial state distribution with an expanded region for the objects of interest."""

    class ExpandedMu(mu_cls):
        def __init__(self, *args, **kwargs):
            self.obj_of_interest_to_region_map = {}
            super().__init__(*args, **kwargs)

        def define_regions(self):
            super().define_regions()

            # Iterate through objects of interest
            for obj in expansion_obj_of_interest:
                for condition in self.init_states:
                    if condition[1] == obj and condition[0] == "On" and condition[2].endswith("init_region"):
                        # Get object's current initial state region
                        region_name = condition[2].replace(self.workspace_name + "_", "")
                        current_range = self.regions[region_name]["ranges"]
                        # Expand the region
                        assert len(current_range) == 1
                        x1, y1, x2, y2 = current_range[0]
                        width = x2 - x1
                        height = y2 - y1
                        new_range = [
                            (
                                x1 - expansion_half_len_factor * width,
                                y1 - expansion_half_len_factor * height,
                                x2 + expansion_half_len_factor * width,
                                y2 + expansion_half_len_factor * height,
                            )
                        ]
                        # Overwrite the initial state region
                        self.regions[region_name]["ranges"] = new_range
                        self.obj_of_interest_to_region_map[obj] = new_range
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    return ExpandedMu


def generate_ood_init_wrapper(expanded_mu_cls, expansion_obj_of_interest, expansion_half_len_factor):
    """
    Given a class with expanded initial regions, generate an environment wrapper that ensures objects of interest
    are in the expanded region.
    """

    class OODInitWrapper(gym.Wrapper):
        global expanded_mu_cls

        def __init__(self, env):
            super().__init__(env)
            self.env = env
            mu = expanded_mu_cls()
            self.obj_of_interest_to_region_map = mu.obj_of_interest_to_region_map

        def reset(self, **kwargs):
            while True:
                print("Resampling objects and fixtures...")
                out = self.env.reset(**kwargs)
                for obj_name in expansion_obj_of_interest:
                    # Make sure at least one object of interest is in the expanded part of its initial region
                    obj_state = self.env.env.object_states_dict[obj_name]
                    obj_x, obj_y = obj_state.get_geom_state()["pos"][:2]
                    if obj_name not in self.obj_of_interest_to_region_map.keys():
                        # This is a rare case where the object of interest does not have
                        # an initial state region defined
                        print("Skipping", obj_name)
                        continue
                    init_range = self.obj_of_interest_to_region_map[obj_name]
                    x1_, y1_, x2_, y2_ = init_range[0]
                    width = (x2_ - x1_) / (1 + 2 * expansion_half_len_factor)
                    height = (y2_ - y1_) / (1 + 2 * expansion_half_len_factor)
                    x1, x2 = (x1_ + x2_) / 2 - width / 2, (x1_ + x2_) / 2 + width / 2
                    y1, y2 = (y1_ + y2_) / 2 - height / 2, (y1_ + y2_) / 2 + height / 2
                    if obj_x < x1 or obj_x > x2 or obj_y < y1 or obj_y > y2:
                        return out

    return OODInitWrapper


def get_expanded_libero_env(
    task, expansion_half_len_factor, ood_only, min_distractors, max_distractors, seed, distractor_seed, resolution=256
):
    """
    Given a LIBERO task, generate an environment with expanded initial regions for the objects of interest
    by a factor of expansion_half_len_factor in each direction. If ood_only is True, *at least one* object
    of interest must be initialized in the expanded part of the region (as opposed to the entire expanded region).

    Distractor objects may be added by setting min_distractors > 0. The number and position of the distractor objects
    is determined for the environment's lifetime by distractor_seed (see comments to generate_mu_with_distractor_objects
    for more details).
    """
    bddl_files_default_path = get_libero_path("bddl_files")

    # Parse the bddl file for this task
    bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    parsed = BDDLUtils.robosuite_parse_problem(bddl_file)

    # Get the scene name
    language = benchmark.grab_language_from_filename(task.bddl_file)
    scene_name = task.bddl_file.replace("_" + language.replace(" ", "_"), "").replace(".bddl", "")

    # Get the mu class
    mu_cls = get_scene_class(scene_name)
    mu_cls_name = mu_cls.__name__
    obj_of_interest = parsed["obj_of_interest"]
    goal_states = parsed["goal_state"]
    goal_states = [tuple(g) for g in goal_states]

    # Make a version of the class with expanded initial regions only for obj_of_interest
    mu_expanded_cls = generate_expanded_mu(mu_cls, obj_of_interest, expansion_half_len_factor)
    if min_distractors > 0:
        mu_expanded_cls = generate_mu_with_distractor_objects(
            mu_expanded_cls, min_distractors, max_distractors, distractor_seed
        )
    mu_expanded_cls.__name__ = mu_cls_name + "Expanded"

    scene_dict = get_scene_dict()
    scene_type = [key for (key, value) in scene_dict.items() if mu_cls in value][0]  # noqa

    task_generation_utils.TASK_INFO = {}

    register_mu(scene_type=scene_type)(mu_expanded_cls)
    print("Registered", mu_expanded_cls.__name__)

    register_task_info(
        language=language,
        scene_name=scene_name + "_EXPANDED",
        objects_of_interest=obj_of_interest,
        goal_states=goal_states,
    )

    # Generate a temp bddl file for the expanded task
    generate_bddl_from_task_info()
    task_bddl_file = "/tmp/pddl/" + scene_name + "_EXPANDED" + "_" + task.language.replace(" ", "_") + ".bddl"

    task_description = task.language
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)

    env.seed(seed)  # Seed will affect object initial positions

    if ood_only:
        ood_init_wrapper_cls = generate_ood_init_wrapper(mu_expanded_cls, obj_of_interest, expansion_half_len_factor)
        env = ood_init_wrapper_cls(env)

    return env, task_description

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size, key="agentview_image"):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[key]
    img = np.flipud(img)
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = Image.fromarray(img)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # resize to size seen at train time
    img = img.convert("RGB")
    return np.array(img)


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, output_dir="out"):
    """Saves an MP4 replay of an episode."""
    rollout_dir = output_dir + f"/rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
