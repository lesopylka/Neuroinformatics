from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import joblib
import pandas as pd
import numpy as np
import os

# База данных
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/beerdb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Beer(Base):
    __tablename__ = "beers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    beer_type = Column(String)
    model_id = Column(Integer, index=True)
    stock = Column(Integer)

Base.metadata.create_all(bind=engine)

# Приложение FastAPI
app = FastAPI(
    title="Beer Recommendation API",
    description="Рекомендуем сорта пива на основе ваших данных и предоставляем информацию о наличии",
    version="1.1",
)

# Загрузка модели
model_path = "model.joblib"
model = joblib.load(model_path)

# Категории
valid_genders = {"м": "Male", "ж": "Female"}
valid_locations = ["Москва", "Питер", "Казань", "Белгород"]

class UserData(BaseModel):
    age: int
    gender: str
    location: str

class BeerInput(BaseModel):
    name: str
    beer_type: str
    model_id: int
    stock: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/recommend/", summary="Получить рекомендации", description="Рекомендует пиво и указывает наличие в базе данных.")
async def recommend_beers(user_data: UserData, db: Session = Depends(get_db)):
    if user_data.gender not in valid_genders:
        raise HTTPException(status_code=400, detail="Некорректный пол. Используйте 'м' или 'ж'.")
    if user_data.location not in valid_locations:
        raise HTTPException(status_code=400, detail=f"Некорректный город. Используйте один из {valid_locations}.")

    gender_encoded = valid_genders[user_data.gender]
    user_input = pd.DataFrame([
        {"age": user_data.age, "gender": gender_encoded, "location": user_data.location}
    ])
    feature_names = ["age", "gender_Male", "gender_Female", "location_Москва", "location_Питер", "location_Казань", "location_Белгород"]
    user_input = pd.get_dummies(user_input, columns=["gender", "location"]).reindex(columns=feature_names, fill_value=0)

    try:
        predictions_proba = model.predict_proba(user_input)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

    top_5_indices = predictions_proba.argsort()[-5:][::-1]
    top_5_beers = [int(idx) for idx in top_5_indices]

    beers_in_db = db.query(Beer).filter(Beer.model_id.in_(top_5_beers)).all()
    recommendations = [
        {
            "beer_id": beer_id,
            "beer_name": f"Пиво {beer_id}",
            "availability": [
                {"name": beer.name, "type": beer.beer_type, "stock": beer.stock}
                for beer in beers_in_db if beer.model_id == beer_id
            ],
        }
        for beer_id in top_5_beers
    ]

    return {"recommendations": recommendations}

@app.post("/add_beer/", summary="Добавить пиво")
async def add_beer(beer: BeerInput, db: Session = Depends(get_db)):
    new_beer = Beer(name=beer.name, beer_type=beer.beer_type, model_id=beer.model_id, stock=beer.stock)
    db.add(new_beer)
    db.commit()
    db.refresh(new_beer)
    return {"message": "Пиво добавлено", "beer": new_beer}

@app.get("/", summary="Главная страница")
async def root():
    return {"message": "Добро пожаловать в Beer Recommendation API! Перейдите на /docs для взаимодействия с API."}
