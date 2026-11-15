# Heart Disease Prediction API

A small FastAPI app that serves predictions from a Random Forest classifier trained on the
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
The app is Dockerized and deployed on Render.

**Live URL:** https://heart-disease-api-fm5c.onrender.com/docs

## Project structure

```
├── app/
│   ├── main.py        # FastAPI app and endpoints
│   └── schemas.py     # Pydantic models (input and output)
├── model/
│   └── heart_model.joblib   # trained model
├── data/
│   └── heart.csv      # dataset from Kaggle
├── train.py           # trains and saves the model
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Endpoints

| Method | Path       | What it does                                   |
|--------|------------|------------------------------------------------|
| GET    | `/health`  | Returns `{"status": "ok"}`                     |
| GET    | `/info`    | Model type and the list of input features      |
| POST   | `/predict` | Takes 13 patient features, returns `heart_disease: true/false` |

## Train the model (one time)

```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
python -m pip install -r requirements.txt
python train.py
```

This creates `model/heart_model.joblib`.

## Run with Docker

```bash
docker-compose build
docker-compose up
```

Open http://localhost:8000/docs

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample.json
```

Response:

```json
{"heart_disease": true, "probability": 0.8}
```

## Deploy on Render

1. Push the project to GitHub.
2. Render > New > Web Service > connect the repo.
3. Language: Docker, root directory empty (the Dockerfile is at the project root).
4. Create the service, wait for the build, then open `/docs`.

The Dockerfile starts uvicorn on the `PORT` variable provided by Render (falls back to 8000 locally).