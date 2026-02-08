# LLM-Guided Pareto Scenario Discovery for Robust Map-Free LiDAR Navigation (Reproducibility Repo)

Bu repo, makaledeki **Hard-NoLLM / Hard-LLM benchmark** üretimi, **hard-bank** derleme,
**robust fine-tuning** ve **figür/tablo üretimi** adımlarını yeniden üretilebilir biçimde
paketlemek için hazırlanmış bir iskelettir.

> Not: LLM bileşeni yalnızca *keşif* sırasında kullanılır. LLM anahtarınız yoksa bile,
> repo içindeki **ön-hesaplanmış artefaktlar** (audit log / manifest / özet sonuçlar)
> üzerinden ana tablo/figürleri yeniden üretebilirsiniz.

---

## 1) Hızlı başlangıç (yalnızca figür/tablo üretimi)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Makaledeki figürleri yeniden üret:**
```bash
python tools/make_figs_robust_results.py --help
```

Örnek (1000 dpi):
```bash
python tools/make_figs_robust_results.py   --eval_dir artifacts/eval_summaries   --out_dir outputs/figs   --dpi 1000
```

> `artifacts/eval_summaries/` klasöründe, makaledeki raporlamayı besleyen örnek/çıktı JSON'ları tutulur.

---

## 2) Yeniden üretilebilirlik seviyeleri

**Seviye A — “Paper artefact replay” (önerilen):**
- `artifacts/` altındaki JSON/JSONL özetlerinden **tablolar ve figürler** üretilir.
- LLM anahtarı / uzun eğitim gerekmez.

**Seviye B — “Evaluation replay”:**
- Paylaşılan senaryo paketleri + (paylaşılan) modeller ile benchmark tekrar çalıştırılır.
- CPU ile çalışır; daha uzun sürer.

**Seviye C — “Full pipeline”:**
- Taban PPO eğitimi + keşif (NoLLM/LLM) + hard-bank + robust fine-tuning + evaluation.
- LLM modunu çalıştırmak için API anahtarı gerekir.

---

## 3) Repo yapısı (önerilen)

- `src/eswa_nav/` : ortam / yardımcı modüller (paket biçimi)
- `tools/`        : çalıştırılabilir betikler (evaluation, finetune, figür üretimi)
- `artifacts/`    : yeniden üretilebilirlik için küçük/orta boy artefaktlar
- `supplementary/`: gerçek-dünya overlay’leri ve ek görseller
- `docs/`         : şema ve çalışma notları
- `paper/`        : (opsiyonel) LaTeX kaynakları

---

## 4) GitHub’a koymanız gereken “asgari” dosyalar (checklist)

### A) Kod (zorunlu)
- **Senaryo DSL + üretim/doğrulama:** `scenario_dsl.py`, senaryo JSON şemaları
- **Simülasyon ortamı:** `ilkkisim.py` (CustomEnv) ve bağımlı yardımcılar
- **Keşif döngüsü (NoLLM + LLM):** discovery scriptleri, coverage tracker, Pareto/elite seçimi
- **Hard-bank derleme + split:** `make_hardbank_manifest.py`, `robust_manifest.json` üretimi
- **Robust fine-tuning:** `run_robust_finetuning.py`
- **Benchmark değerlendirme:** `evaluate_*` betikleri, `run_eval_manifest.py`
- **Analiz + figür üretimi:** `make_figs_*.py`, tablo üretimi ve CI hesapları

### B) Konfigürasyon (zorunlu)
- `requirements.txt` / `environment.yml`
- `run_config.json` (eğitim/eval parametreleri)
- `split_salt` ve seed listesi (42–51)
- CLI komutları (`scripts/*.sh`) veya Makefile

### C) Artefaktlar (en azından “paper replay” için)
- `robust_manifest.json` (dosya listeleri + split)
- LLM audit izleri: `llm_audit_log.jsonl` (tamamı büyükse örnek + checksum)
- Benchmark/eval özetleri: `*_ep3000.json` gibi dosyalar
- (opsiyonel) Pareto arşivleri, coverage snapshot’ları

### D) Modeller
- En azından **Base-100k**: `ppo_static_shaping_model.zip`, `ppo_static_shaping_vecnorm.pkl`
- Robust modeller çok büyükse GitHub Releases / Zenodo üzerinden sunulabilir.

---

## 5) Lisans & Atıf
- Kod için `LICENSE`
- Atıf için `CITATION.cff`

---

## İletişim
Sorun/eksik için “Issues” açabilirsiniz.
