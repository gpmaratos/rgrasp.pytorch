def visualize(visualizer, cfg):
    for ind, inp in enumerate(visualizer):
        iarr, bboxes = inp
        visualizer.show_ground_truth(iarr, bboxes)
