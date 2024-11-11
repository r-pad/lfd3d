import torch


def rmse_pcd(x, y, mask):
    """
    Calculate the RMSE over the predicted point cloud
    and the ground truth point cloud. Returns RMSE averaged
    over all points. Assumes input of shape (N, D, 3)

    x: Predicted pcd
    y: Actual pcd
    mask: Mask of points to be considered
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3

    rmse = []
    for pcd_x, pcd_y, mask in zip(x, y, mask):
        pcd_x, pcd_y = pcd_x[mask], pcd_y[mask]
        rmse.append(((pcd_x - pcd_y) ** 2).mean() ** 0.5)
    rmse = torch.stack(rmse)
    return rmse


def chamfer_distance(x, y, mask):
    """
    Calculate the Chamfer distance over the predicted point cloud
    and the ground truth point cloud. Returns Chamfer Distance averaged
    over all points. Assumes input of shape (N, D, 3)

    x: Predicted pcd
    y: Actual pcd
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3

    chamfer_dist = []
    for pcd_x, pcd_y, mask in zip(x, y, mask):
        pcd_x, pcd_y = pcd_x[mask], pcd_y[mask]

        # Compute squared distances from each point in x to the closest point in y
        dist_x_to_y = torch.cdist(pcd_x, pcd_y)  # Pairwise distances between all points
        # Min distance for each point in x
        min_dist_x_to_y = torch.min(dist_x_to_y, dim=1)[0]

        # Compute squared distances from each point in y to the closest point in x
        dist_y_to_x = torch.cdist(pcd_y, pcd_x)  # Pairwise distances between all points
        # Min distance for each point in y
        min_dist_y_to_x = torch.min(dist_y_to_x, dim=1)[0]

        # Chamfer distance is the average of the two distances
        chamfer = torch.mean(min_dist_x_to_y) + torch.mean(min_dist_y_to_x)
        chamfer_dist.append(chamfer)
    chamfer_dist = torch.stack(chamfer_dist)
    return chamfer_dist
