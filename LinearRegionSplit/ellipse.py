import numpy as np
import numexpr as ne

def _get_zero_centered_circle(radius: float, num_samples: int, split_num: int, index: int) -> np.ndarray:
  """Get a zero-centered circle of a given radius.
  Args:
    radius: radius of the circle.
    num_samples: number of points to sample on the circle.
  Returns:
    A numpy array of circle points.
  """
  assert num_samples % split_num == 0
  each = num_samples // split_num
  rads = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
  rads = rads[index * each: (index + 1) * each]
  circle = np.zeros([each, 2])
  ne.evaluate('cos(rads)', out=circle[:, 0])
  ne.evaluate('sin(rads)', out=circle[:, 1])
  circle *= radius
  return circle

def _get_2d_points_fitting_transform(vertices: np.ndarray, 
                                     scale: float) -> np.ndarray:
  """Get a matrix transformation projecting circle points onto vertices.
  Args:
    vertices: resulting points of the transformation.
    scale: scale (radius) of the input equidistant circle points.
  Returns:
    Transformation matrix.
  """
  # angles = np.pi * np.linspace(0, 2, len(vertices), endpoint=False)
  angles = np.array([np.pi / 3, np.pi, np.pi * 5 / 3])
  angles = np.reshape(angles, (-1, 1))
  coords_2d = np.concatenate((np.cos(angles), np.sin(angles)), axis=1) * scale
  transform = np.dot(np.linalg.pinv(coords_2d), vertices)
  return transform


def get_fitted_ellipse_and_stats(num_samples: int, \
            vertices: np.ndarray, split_num: int, index: int) -> np.ndarray:
  """Get an ellipse passing through given vertices.
  The ellipse is centered at the mean of the vertices;
  Args:
    num_samples: number of points sampled on the circle.
    vertices: vertices to fit the circle to.
  Returns:
    A numpy array of points on a circle and dict with statistics.
  """
  shape = vertices[0].shape if vertices.ndim > 1 else (1,)
  vertices = np.reshape(vertices, (vertices.shape[0], -1))

  center = np.mean(vertices, axis=0)
  circle = _get_zero_centered_circle(1, num_samples, split_num, index)

  transform = _get_2d_points_fitting_transform(vertices - center, scale=1)
  circle = np.dot(circle, transform)
  circle += center
  circle = np.reshape(circle, (num_samples // split_num,) + shape)
  return circle
