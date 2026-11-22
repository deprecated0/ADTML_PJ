# ADTML_PJ

Учебный ML-проект: предсказание дефолта по кредитке (датасет с UCI).

## Что внутри

- **данные** загрузка, чистка, валидация через Great Expectations
- **фичи** агрегаты по истории платежей, утилизация лимита, биннинг возраста
- **модель** sklearn Pipeline (GBC + LogReg), подбор через GridSearchCV
- **эксперименты** прогоны в MLflow
- **версионирование** DVC (2 стадии: prepare_data, train_model)
- **тесты** pytest + GitHub Actions (flake8, black, GE-валидация)
- **сервис** FastAPI, `/predict` отдаёт класс и вероятность
- **мониторинг** PSI-скрипт сравнивает «новые» данные с трейном

## Как запустить

```bash
git clone https://github.com/deprecated0/ADTML_PJ.git && cd ADTML_PJ
pip install -r requirements.txt

# данные и модель через DVC
dvc repro

# API
uvicorn src.serve:app --reload

make docker