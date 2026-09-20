import sys
sys.path.insert(0, '20_model_compression/python')
from demo import simulate_distillation
for n in [200, 300, 400, 500]:
    for t in [2.0, 4.0, 8.0]:
        m = simulate_distillation(n_samples=n, n_classes=5, n_features=8, temperature=t)
        lift = m['student_distilled_acc'] - m['student_independent_acc']
        print(f'n={n} T={t}: teacher={m["teacher_acc"]:.3f} indep={m["student_independent_acc"]:.3f} distill={m["student_distilled_acc"]:.3f} lift={lift:+.3f}')