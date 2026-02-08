# Artefaktlar ve Şemalar

Bu repo iki tip çıktı üretir:

## 1) Senaryo tanımları (JSON)
Her senaryo dosyası tek bir ortamı tanımlar:
- başlangıç/goal
- AABB engeller
- sensör bozulmaları (`sensor_cfg`)

Kaynak: `scenario_dsl.py`

## 2) Değerlendirme özetleri (JSON / JSONL)
Makaledeki tablolar/figürler, değerlendirme scriptlerinin ürettiği özetlerden türetilir.

Önerilen minimum alanlar:
- `success` / `collision` / `timeout` (bool veya 0/1)
- `return` ya da `avg_return`
- opsiyonel: `steps`, `min_lidar`, `scenario_id`

Eğer dosya formatı farklıysa, `tools/make_figs_robust_results.py` içinde yer alan
yükleyici/parsing bölümü genişletilerek uyarlanabilir.
