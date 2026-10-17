"""Reproducible Russian interpretation, all five seeds and fixed thresholds."""
from pathlib import Path
import json,sys
import numpy as np
from tabulate import tabulate

OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT.parents[1]))
from experiments.exp1_encoder_tautology import _safe_json
r=json.loads((OUT/'exp3_elliptic_results.json').read_text())
def rows(field,**kw):return [x for x in r[field] if all(x.get(k)==v for k,v in kw.items())]
def nums(xs,key):return np.asarray([np.nan if x[key] is None else x[key] for x in xs],float)
def fmt(v):return f'{v:.6g}' if np.isfinite(v) else 'undefined'
def table(xs):return tabulate(xs,headers='keys',tablefmt='github',floatfmt='.6g')
models=r['protocol']['models'];passed=sum(c['passed'] for c in r['checks'])
law=[];phase=[];iso=[];h3=[];probes=[];calibration=[]
for model in models:
    xs=rows('frequency_runs',model=model,split='test_id_ellip',regime='T3',method='S3b_diff')
    err=nums(xs,'slope_abs_error');r2=nums(xs,'r2')
    law.append(dict(model=model,mean_slope=nums(xs,'slope').mean(),mean_abs_error=err.mean(),worst_abs_error=err.max(),mean_R2=r2.mean(),min_R2=r2.min(),invalid=sum(x['invalid'] for x in xs)))
    ps=nums(rows('phase_runs',model=model),'mean_phase_gap')
    phase.append(dict(model=model,mean_gap=ps.mean(),max_gap=ps.max(),min_gap=ps.min()))
    for a in [.7,1.2,1.8]:
        xs=rows('iso_runs',model=model,a=a);qs=np.asarray([[np.nan if v is None else v for v in x['q']] for x in xs])
        iso.append(dict(model=model,a=a,mean_std_log_omega=nums(xs,'cv_log_omega').mean(),max_std_log_omega=nums(xs,'cv_log_omega').max(),mean_q=qs.mean(),worst_abs_q_minus1=abs(qs-1).max()))
    for split in ['test_id_ellip','test_ood_ecc','test_ellip_peri','test_ellip_apo']:
        xs=rows('frequency_runs',model=model,split=split,regime='T3',method='S3b_diff');qs=np.asarray([[np.nan if v is None else v for v in x['q']] for x in xs])
        calibration.append(dict(model=model,split=split,mean_slope=nums(xs,'slope').mean(),mean_R2=nums(xs,'r2').mean(),mean_q=qs.mean(),min_q=qs.min(),max_q=qs.max(),worst_abs_q_minus1=abs(qs-1).max(),invalid=sum(x['invalid'] for x in xs)))
for model in ['M0_Koopman','M1_Ellip_Inv']:
    for method in ['S3b_std','S3b_diff','S3a_diff','S2']:
        xs=rows('frequency_runs',model=model,split='test_id_ellip',regime='T1',method=method)
        h3.append(dict(model=model,method=method,mean_abs_error=nums(xs,'slope_abs_error').mean(),worst_abs_error=nums(xs,'slope_abs_error').max(),mean_R2=nums(xs,'r2').mean(),min_R2=nums(xs,'r2').min(),invalid=sum(x['invalid'] for x in xs)))
for model in models:
    for split in ['test_id_ellip','test_ood_ecc']:
        for feature,method in [(f,m) for f in ['norm','state','log_omega'] for m in ['ridge','poly2']]:
            entry=dict(model=model,split=split,feature=feature,method=method)
            for target in ['log_r0','log_v0','log_a','e','log_h','log_abs_energy']:
                xs=rows('probe_runs',model=model,split=split,feature=feature,method=method,target=target)
                entry[target]=nums(xs,'r2').mean()
            probes.append(entry)
circ=np.mean(nums(rows('phase_runs',model='M1_circ'),'mean_phase_gap'))
inv=np.mean(nums(rows('phase_runs',model='M1_Ellip_Inv'),'mean_phase_gap'))
norm_rows=rows('probe_runs',model='M1_Ellip_Inv',split='test_id_ellip',feature='norm',method='ridge',target='log_a')
energy_values=nums(rows('probe_runs',model='M1_Ellip_Inv',split='test_id_ellip',feature='log_omega',method='poly2',target='log_abs_energy'),'r2')
summary=dict(law=law,phase=phase,iso=iso,H3=h3,probes=probes,calibration=calibration)
reference=json.loads((OUT/'task4_h3_reference.json').read_text())['records']
reference_table=[]
for label in ['identity_physical_context','oracle_mean_anomaly_circle']:
    for method in ['S3b_std','S3b_diff','S3a_diff','S2']:
        xs=[x for x in reference if x['reference']==label and x['method']==method]
        reference_table.append(dict(reference=label,method=method,worst_abs_error=nums(xs,'slope_abs_error').max(),min_R2=nums(xs,'r2').min()))
summary['H3_calibration']=reference_table
(OUT/'task4_summary.json').write_text(json.dumps(_safe_json(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
lines=[
'# Task 4 — эллиптические орбиты и инварианты',
f"Полный эксперимент завершён: 20 GPU-обучений × 2000 шагов, batch 256, пять сидов 42–46. Научная приёмка: **{passed}/{len(r['checks'])} PASS**; exit {'0' if r['acceptance_passed'] else '1'}. Завершение вычислений не означает подтверждение всех гипотез ТЗ.",
f"Единственный FAIL — ожидание R²(log a из нормы M1Inv)<0.75 на каждом сиде. Фактически: {', '.join(str(x['seed'])+': '+fmt(x['r2']) for x in norm_rows)}. Четыре из пяти значений выше порога. Следовательно, норма сохраняет заметную информацию о полуоси; заявленное жёсткое разделение «норма — только локальная кинематика, частоты — энергия» данными не подтверждается. При этом энергетическая проба от log-частот даёт R² от {fmt(energy_values.min())} до {fmt(energy_values.max())} и проходит свой критерий.",
'Перенос на экстремальные e пока ненадёжен: в OOD_ecc среднее R² закона T3 составляет ≈0.920 для M1Inv и ≈0.726 для M2Inv, худшая относительная ошибка частоты — ≈79% и практически 100%. Это не входит в заданную приёмку закона на ID, но существенно ограничивает положительный научный вывод. В одном OOD случае M2 q≈2.46e−14 формально остаётся положительным, поэтому invalid=0 не означает корректный период.',
'Для проверки начните с этого разбора, затем откройте [четырёхпанельный график](figures/exp3_elliptic_kepler.png) и [полные таблицы](exp3_elliptic_report.md). [JSON](exp3_elliptic_results.json) содержит каждую орбиту, прогнозы проб, параметры нормализации и коэффициенты; [протокол](protocol_task4.md) фиксирует решения до прогона. Чекпойнты находятся в checkpoints/.',
'## Приёмка',table(r['checks']),
'Все пять сидов включены в агрегаты. Невалидные частоты не удаляются. Пороговые значения после получения результатов не менялись.',
'## Блок 1 и фазовая инвариантность',table(phase),
f'Среднее расхождение M1circ / M1Inv: {fmt(circ)} / {fmt(inv)}; отношение средних — {fmt(circ/inv)}. Это тест одной и той же физической орбиты с двумя стартами, поэтому истинная частота совпадает. Различающиеся предсказания показывают нарушение фазовой инвариантности.',
'Такое нарушение опровергает обобщение круговой модели на эти эллипсы, но само по себе не доказывает единственную конкретную эвристику нормы. Возможны смешивание кинематики и интегралов, недостаточный охват обучающих данных и ошибки представления/оценки частоты. Отдельно полезно сопоставить OOS-пробы и абсолютную калибровку ниже.',
'## Блок 2: H3 на истинной закодированной траектории T1',table(h3),
'H3 по критериям ТЗ подтверждена: у M0K оба фазовых метода проходят на всех пяти сидах, у M1Inv — S3b_std. S3b_diff у M1Inv не проходит все строгие пороги H3; этот результат не скрывается и не заменяется выбором метода отдельно по сидам. DMD деградирует в обоих случаях.',
'Утверждение о топологическом обороте фазы на полный период слабее требования почти идеальной регрессии частоты на конечном окне. PCA, вычитание центра, гармоники и неполные периоды могут влиять на оценку. Поэтому отрицательный H3 означает провал именно заявленного протокола и порогов; он не доказывает бесполезность фазовых оценивателей вообще. Метод std/diff не выбирался отдельно по сидам.',
'Дополнительная калибровка H3 использует те же 512 временных точек: identity отображает физическое окно 16×4 в 64 координаты; oracle строит идеальную окружность по истинной средней аномалии. Oracle является только контрольным сигналом для оценивателя и никогда не участвует в обучении или пробах. Эта диагностика не входит в исходную приёмку.',
table(reference_table),
'Калибровка использует точные параметры орбит из сохранённого GPU-JSON. Различие std/diff у M1Inv нельзя целиком объяснить самим генератором эллипсов или длиной ряда: существенна геометрия конкретного представления. Oracle S3b_diff восстанавливает частоту до машинной точности, а FFT сохраняет дискретизационную погрешность; количественные значения обеих калибровок приведены выше.',
'## Блок 2: закон Кеплера в автономном T3',table(law),
'Наклон −1.5 и высокий R² проверяют степенную зависимость. Они не фиксируют множитель частоты: модель с omega_hat = c*a^(-1.5) проходит эти две проверки при любом c>0. Ниже отдельно приведён q=omega_hat/omega_true, включая худшую ошибку; новых критериев приёмки задним числом не добавлено.',
table(calibration),
'## Блок 3: изоэнергетические группы',table(iso),
'std(log omega) вычислена внутри каждого a по 15 эксцентриситетам. Здесь e и стартовая фаза меняются одновременно: тест оценивает их совместную устойчивость, а не чистый эффект e при фиксированной фазе. Небольшая дисперсия при неверном mean q означает устойчивую, но неверно откалиброванную частоту.',
'## Блок 4: OOS-пробы',table(probes),
'В этой сводке показаны обе пробы для всех трёх признаков. Панель (d) использует заранее выбранные norm/Ridge и state/logomega/poly2. Все преобразования и коэффициенты обучены исключительно на train_ellip, до применения к ID и OOD_ecc. Отрицательные R² сохранены и означают, что прогноз хуже константы среднего данной тестовой выборки.',
'Пробы имеют существенную зависимость от сложности: у M1Inv на ID e декодируется линейно из log-частот со средним R²≈0.640, но poly2 даёт ≈−4.054. Значит, провал этой нелинейной пробы не доказывает отсутствия информации об e. Poly2 даёт 560 признаков для 32 частот и 2144 для полного z при всего 200 обучающих орбитах; фиксированная Ridge(.01) не гарантирует хорошего обобщения. На OOD_ecc деградация гораздо сильнее. Следующий этап должен сравнить регуляризацию через train-only cross-validation и размер обучающей выборки, без подбора по ID/OOD результатам.',
'Высокая декодируемость log a из частот является свидетельством доступности информации об энергии, но не доказательством того, что предиктор реализует физическую формулу энергии. log|E| = -log a - log 2 при mu=1, поэтому эти две цели дают аффинно эквивалентную проверку. Сферический M3 имеет практически постоянную норму: нормовая проба корректно сводится к intercept, а не получает информацию из численного шума.',
'## Ограничения постановки и дальнейшие действия',
'1. H=256 покрывает 10.24 единиц времени при периоде до 70.25 в train. Две независимые точки из доступной 21-единичной записи не гарантируют перицентр и апоцентр. Для проверки полного цикла нужен отдельный заранее заданный эксперимент с более длинными записями.',
'2. Short и Inv различаются одновременно горизонтом, числом стартов и частотным штрафом. Для причинной атрибуции улучшения нужна факторная абляция: long/single, long/paired/no-inv, short/paired/inv при сопоставимом вычислительном бюджете.',
'3. Инвариантность не исключает постоянную частоту для всех орбит. Поэтому нужны одновременно M0-контроль, восстановление зависимости по a, абсолютный q и OOS-пробы.',
'4. Изоэнергетический тест стоит расширить на фиксированные фазы и ориентации при изменении только e, затем отдельно менять фазу. OOD по e не заменяет OOD по a, mu или возмущениям закона силы.',
'5. Круговое движение в латентном пространстве совместимо с неравномерным движением по физическому эллипсу. Для более сильного утверждения о динамике нужны декодированная физическая траектория, дрейф E/h и закон площадей; одних частот и проб недостаточно.',
'6. Эти результаты относятся к данной контролируемой реализации и протоколу. Они не являются автоматическим опровержением всех архитектур JEPA или всех утверждений JEPA-Anything.',
'7. Нормовое сопоставление с физическим окном из ТЗ создаёт потенциальное напряжение целей: RMS физического окна зависит от фазы эллипса, тогда как ортогональный латентный поток сохраняет полную норму. Штраф частот может отделять энергию от этой переменности, но точное совпадение всех латентных целей не гарантировано. Отдельная абляция нормового сопоставления нужна для проверки этого механизма.',
'8. Seed меняет одновременно инициализацию модели и случайные параметры train/test орбит. Межсидовая дисперсия смешивает оба источника вариации; для их раздельной оценки нужен фиксированный тестовый набор и перекрёстный дизайн model-seed × data-seed.',
'## Воспроизводимость и проверка',
f"Время полного прогона: {r['elapsed_seconds']:.1f} с. Среда и GPU — в JSON и [preflight](gpu_preflight.json). Выводы тестов: [локально](unit_tests_local.txt), [IDH](unit_tests_idh.txt). [stdout](exp3_elliptic_stdout.txt).",
'[Независимый аудит](provenance/audit_task4_results.json) проверяет сохранность прежних отчётов, SHA256 артефактов/исходников, 20 CUDA-чекпойнтов, пересчёт наклонов, фазовых отношений и OOS-прогнозов по сохранённым коэффициентам. Исходники GPU-прогона сохранены в provenance/task4_sources.zip; оригинальные исходники Task 3.2 также архивированы.',
'После прогона выполнена только правка отображения: единый цвет каждой модели во всех оценивателях панели (a). [Квитанция правки](provenance/presentation_changes.json) фиксирует хеш исходного/итогового графика и неизменность числового JSON. Оригинальный GPU-график сохранён отдельно; исходники GPU-прогона остаются в неизменном ZIP.',
'Для M0 недостаточно одного seed при переносе между версиями PyTorch: побитовая инициализация локально и на IDH различается. В provenance/frozen_controls/ сохранены все пять наборов весов, восстановленных в исходной среде и проверенных по исходным before/after SHA256. Аудит использует эти точные веса, не новую локальную инициализацию. Квитанции экспорта и скачивания сохранены; никакого обучения M0 или изменения результатов не выполнялось.',
'Обнаружена и межплатформенная неоднозначность e-сетки: выбор минимума абсолютной корреляции среди перестановок с почти равными оценками чувствителен к округлению BLAS. Внутри зафиксированной GPU-среды сетка детерминирована и одинакова по сидам; локальная среда может выбрать другую почти равноценную перестановку. Поэтому независимый аудит и H3-калибровка воспроизводят именно сохранённые a/e/M0/ориентации из GPU-JSON через analytical_ellipse_batch, а не повторяют выбор перестановки. Это ограничение переносимости seed-протокола явно зафиксировано; GPU-результаты и критерии не менялись. Для будущего протокола следует фиксировать саму перестановку или выбирать её по точной целочисленной ранговой корреляции.',
'![Четыре панели](figures/exp3_elliptic_kepler.png)']
(OUT/'task4_analysis.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
print('Analysis generated:',OUT/'task4_analysis.md')
