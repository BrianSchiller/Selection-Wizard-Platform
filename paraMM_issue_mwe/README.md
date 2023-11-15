0. Optionally first create an environment with `python3 -m venv venv` and activate it with `source venv/bin/activate`
1. Run `pip install -r requirements.txt`
2. Run `./mwe.py` twice to see the different output despite the fixed seed
3. I also note that it does not seem to matter how the seed is set. E.g., instead setting it with `np.random.seed(seed)` has the same effect.
