"""
Constants for the scene understanding pipeline.
Magic numbers, thresholds, and lookup tables.
"""

# RAM++ generic tags to filter out
RAMPP_GENERIC_TAGS = {
    "object", "objects", "thing", "things", "item", "items",
    "entity", "entities", "scene", "image", "photo", "picture",
}

# Florence-2 stopwords for label extraction from captions
FLORENCE2_CAPTION_STOPWORDS = {
    # articles / determiners
    "a", "an", "the", "some", "one", "two", "three",
    # prepositions / conjunctions
    "with", "on", "of", "in", "at", "by", "and", "or", "from", "to",
    "for", "as", "up", "out", "into", "over", "under", "about", "around",
    # verbs / auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "can", "may", "will", "appears", "seems", "showing", "shows", "shown",
    # pronouns
    "this", "that", "these", "those", "it", "its", "there", "their",
    # meta / photographic words
    "image", "photo", "picture", "view", "close", "shot",
    # positional / descriptive (these are adjectives, not nouns)
    "side", "top", "front", "back", "left", "right", "center", "middle",
    # common adjectives that precede the actual noun
    "red", "blue", "green", "yellow", "white", "black", "brown", "grey",
    "gray", "orange", "purple", "pink", "dark", "light", "bright",
    "large", "small", "big", "little", "tiny", "tall", "short", "long",
    "old", "new", "open", "closed", "empty", "full", "flat", "round",
    "square", "wooden", "metal", "plastic", "glass", "stone", "brick",
    "single", "double", "multiple", "various", "different", "same",
    # filler adverbs
    "very", "quite", "just", "also", "well",
}

# Spatial predicates for relation extraction
SPATIAL_PREDICATES = [
    "left_of", "right_of", "above", "below", "in_front_of", "behind",
    "inside", "outside", "on", "under", "next_to", "touching", "overlapping"
]

# Default Grounding DINO query
DEFAULT_GDINO_QUERY = (
    "person. animal. vehicle. furniture. appliance. food. "
    "clothing. container. tool. building. plant. electronics. object."
)

# Depth colormap
DEPTH_COLORMAP = "inferno"  # OpenCV colormap name

# Default camera FOV estimates (degrees)
DEFAULT_HORIZONTAL_FOV = 71.0  # iPhone/typical smartphone
DEFAULT_VERTICAL_FOV = 55.0

# Image resizing parameters
DEFAULT_MAX_IMAGE_SIDE = 1280
DEFAULT_INTERPOLATION = "area"  # cv2.INTER_AREA for downscaling

# SAM2 AMG default parameters
SAM2_AMG_DEFAULTS = {
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.95,
    "points_per_side": 32,
    "points_per_batch": 32,
    "min_mask_region_area": 200,
}

# Grounding DINO thresholds
GDINO_DEFAULTS = {
    "box_threshold": 0.3,
    "text_threshold": 0.25,
}

# Pix2SG relation thresholds
PIX2SG_DEFAULTS = {
    "mask_overlap_thresh": 0.05,
    "relation_min_mask_overlap": 0.05,
    "relation_bbox_touch_margin_px": 2,
    "depth_near_threshold": 1.0,
    "depth_far_threshold": 3.0,
    "max_relations_per_object": 8,
}

# Depth mask matching thresholds
DEPTH_MASK_DEFAULTS = {
    "match_iou_thresh": 0.1,
    "adaptive_erosion": True,
}

# RAM++ defaults
RAMPP_DEFAULTS = {
    "image_size": 384,
    "vit": "swin_l",
    "default_confidence": 0.70,
    "max_tags": 8,
}

# Florence-2 defaults
FLORENCE2_DEFAULTS = {
    "model_id": "microsoft/Florence-2-large",
    "confidence": 0.75,
}

# Output paths (relative to output directory)
OUTPUT_PATHS = {
    "depth_dir": "depth",
    "scene_graph_dir": "scene_graph",
    "masks_dir": "scene_graph/masks",
    "depth_mask_dir": "scene_graph/depth_mask",
}

# File naming patterns
FILE_PATTERNS = {
    "depth_npy": "{stem}_depth_metric.npy",
    "depth_png": "{stem}_depth_map.png",
    "segmentation_png": "{stem}_segmentation.png",
    "tinted_overlay_png": "{stem}_tinted_overlay.png",
    "depth_mask_mapping_png": "{stem}_depth_mask_mapping.png",
    "scene_json": "{stem}_scene.json",
}
