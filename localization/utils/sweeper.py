from tqdm import tqdm

def tupify(d: dict):
  """
  Make all values in a dictionary 1-element tuples.
  """
  return { k: (v,) for k, v in d.items() }

def sweep_func(func, kwargs_array):
  # Run the function for each set of kwargs
  output = []
  all_kwargs = []
  for kwargs in tqdm(kwargs_array):
      output.append(func(**kwargs))
      all_kwargs.append(kwargs)
  return output, all_kwargs