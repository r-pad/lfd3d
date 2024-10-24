import torch


def rmse_pcd(x, y):
    """
    Calculate the RMSE over the predicted point cloud
    and the ground truth point cloud. Returns RMSE averaged
    over all points. Assumes input of shape (N, D, 3)

    x: Predicted pcd
    y: Actual pcd
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    rmse = ((x - y) ** 2).mean(axis=(1, 2)) ** 0.5
    return rmse


def chamfer_distance(x, y):
    """
    Calculate the Chamfer distance over the predicted point cloud
    and the ground truth point cloud. Returns Chamfer Distance averaged
    over all points. Assumes input of shape (N, D, 3)

    x: Predicted pcd
    y: Actual pcd
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3

    # Compute squared distances from each point in x to the closest point in y
    dist_x_to_y = torch.cdist(x, y)  # Pairwise distances between all points
    min_dist_x_to_y = torch.min(dist_x_to_y, dim=2)[
        0
    ]  # Min distance for each point in x

    # Compute squared distances from each point in y to the closest point in x
    dist_y_to_x = torch.cdist(y, x)  # Pairwise distances between all points
    min_dist_y_to_x = torch.min(dist_y_to_x, dim=2)[
        0
    ]  # Min distance for each point in y

    # Chamfer distance is the average of the two distances
    chamfer_dist = torch.mean(min_dist_x_to_y, dim=1) + torch.mean(
        min_dist_y_to_x, dim=1
    )

    return chamfer_dist
