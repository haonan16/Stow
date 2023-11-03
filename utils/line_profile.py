from line_profiler import LineProfiler

def profile_func(func, args):
    # Instantiate the LineProfiler
    lp = LineProfiler()
    # Add functions to the profiler
    lp.add_function(func)
    # Enable the profiler
    lp.enable_by_count()

    func(*args)

    # Disable the profiler
    lp.disable_by_count()

    # Print the profiling results
    lp.print_stats()
