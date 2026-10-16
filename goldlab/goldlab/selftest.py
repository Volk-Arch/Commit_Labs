"""САМОПРОВЕРКА: прогоняет опыты пресетом «как в документе» (или «быстро» с --quick) и сверяет с эталонами из документов.
Эталоны сняты при параметрах документа; в быстром режиме окна короче и часть чисел законно уходит в ≈/✗ (разрешение спектра).
Запуск: python -m goldlab.selftest [--quick] [--only id1,id2]. Каждый прогон сохраняется в runs/ (тег selftest)."""
import sys, time
from . import experiments as E, store

ORDER = ['world', 'clock', 'light_point', 'bell', 'slice_rt', 'nonabelian', 'dispersion', 'charge_force', 'atom11', 'hydrogen11',
         'light31', 'coulomb31', 'geom_charge', 'hydrogen31', 'color', 'pair',
         'onebody_long', 'dirac_walk', 'field11', 'onerule', 'pauli_atom', 'helium_field', 'lithium_scf', 'whytau', 'nonull', 'lightspin', 'orderatom', 'heliumorder', 'pairexchange', 'pairnorm', 'vectorvertex', 'lithiumnew']
QUICK_SKIP = {'pair', 'lithium_scf', 'pairexchange', 'pairnorm', 'vectorvertex', 'lithiumnew'}                       # долгий (GMRES); в быстром режиме пропускается

def main():
    full = '--quick' not in sys.argv; only = None
    if '--only' in sys.argv: only = sys.argv[sys.argv.index('--only') + 1].split(',')
    preset = 'как в документе' if full else 'быстро'; total = {'ok': 0, 'near': 0, 'off': 0}; T0 = time.time()
    print(f'самопроверка goldlab, пресет «{preset}»' + ('' if full else ' — эталоны документа сняты на длинных окнах, ≈/✗ здесь ожидаемы у спектров'), flush=True)
    for eid in ORDER:
        if only and eid not in only: continue
        if not full and not only and eid in QUICK_SKIP: print(f'  — {eid}: пропущен (долгий; --only {eid})', flush=True); continue
        e = E.REG[eid]; pr = dict(e['presets'].get(preset, {}))
        try:
            r = E.run(eid, pr); store.save(r, 'selftest')
        except Exception as ex:
            print(f'  ✗ {eid}: ошибка {type(ex).__name__}: {ex}', flush=True); total['off'] += 1; continue
        marks = []
        for n in r['numbers']:
            if n['status']:
                total[n['status']] += 1; sym = {'ok': '✓', 'near': '≈', 'off': '✗'}[n['status']]
                marks.append(f"{sym} {n['label']}: {n['value']:.4g} (эталон {n['expected']:.4g})" if n['value'] is not None else f"{sym} {n['label']}: —")
        print(f"  {eid} ({r['seconds']} с): " + ('; '.join(marks) if marks else 'без эталонов'), flush=True)
    print(f"итог: ✓ {total['ok']}, ≈ {total['near']}, ✗ {total['off']}; {time.time() - T0:.0f} с", flush=True)

if __name__ == '__main__':
    main()
