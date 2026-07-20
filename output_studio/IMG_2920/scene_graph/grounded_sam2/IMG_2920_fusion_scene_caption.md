# Fusion Scene Caption - grounded_sam2

You are a scene-grounded captioner and verifier.

Use ALL provided artifacts jointly:
- Original image: /private/var/folders/_k/s8lgfj6d7k56717h6w7c6qw80000gn/T/gradio/a8e94ee7960f78f1b3e96cb7da29540fb2987b260f032b2887d8e8258f910966/IMG_2920.JPG
- Track: grounded_sam2
- scene_graph JSON: scene_graph/grounded_sam2/IMG_2920_scene.json
- relations JSON: scene_graph/grounded_sam2/IMG_2920_relations.json
- layers JSON: scene_graph/grounded_sam2/IMG_2920_layers.json
- mask hierarchy JSON: scene_graph/grounded_sam2/IMG_2920_mask_hierarchy.json
- depth_mask_A JSON: scene_graph/grounded_sam2/IMG_2920_depth_mask_A.json
- depth_mask_B JSON: 
- segmentation image: scene_graph/grounded_sam2/IMG_2920_sam2_segmentation.png
- tinted overlay: scene_graph/grounded_sam2/IMG_2920_sam2_tinted_overlay.png
- relations map image: scene_graph/grounded_sam2/IMG_2920_relations_map.png
- layers image: scene_graph/grounded_sam2/IMG_2920_layers.png
- mask hierarchy image: scene_graph/grounded_sam2/IMG_2920_mask_hierarchy.png
- regions JSON: scene_graph/grounded_sam2/IMG_2920_regions.json
- regions index image: scene_graph/grounded_sam2/IMG_2920_regions.png
- regions overlay image: scene_graph/grounded_sam2/IMG_2920_regions_overlay.png
- region segmentation image (parallel): scene_graph/grounded_sam2/IMG_2920_region_segmentation.png
- region SAM2-style labelled segmentation (parallel): scene_graph/grounded_sam2/IMG_2920_region_sam2_style_segmentation.png
- region tinted overlay (parallel): scene_graph/grounded_sam2/IMG_2920_region_tinted_overlay.png

Goals:
1) Write a detailed caption (10-16 sentences) grounded in BOTH image evidence and scene-graph evidence.
2) Include object attributes, spatial relations, depth/layer cues, and part-whole hierarchy when supported.
3) Avoid hallucinations: do not mention entities absent from both image and graph.
4) If evidence conflicts, explicitly flag uncertainty and alternatives.

