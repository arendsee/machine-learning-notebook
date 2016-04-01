#!/usr/bin/env python3
# WARNING There is currently a bug in pypge
from pypge.search import PGE
from pypge.benchmarks import explicit
from pypge import fitness_funcs as FF
import sympy

prob = explicit.Lipson_02()

pge = PGE(
    system_type="explicit",
    search_vars="y",
    usable_vars=prob['xs_str'],
    usable_funcs=[sympy.exp, sympy.cos, sympy.sin, sympy.Abs],
    pop_count=3,
    peek_count=9,
    peek_npts=100,
    max_iter=6,
    print_timing=True,
    log_details=True,
    fitness_func=FF.normalized_size_score
    )

pge.fit(prob['xpts'], prob['ypts'])

final_paretos = pge.get_final_paretos()
final_list = [item for sublist in final_paretos for item in sublist]

for best_m in final_paretos[0]:
    print(best_m)
