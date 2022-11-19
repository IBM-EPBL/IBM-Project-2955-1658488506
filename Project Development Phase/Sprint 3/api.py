from fastapi import FastAPI,HTTPException,status,BackgroundTasks
from utils.auth_utils import genToken, verifyToken
from db.db_enum import Queries
from db.db_helpers import *
from models.fast_api_models import *
from models.models import *
from prediction.prediction import *
import uvicorn

app=FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


#add more routes

#Admin routes 


#Doctor Routes
@app.post('/doctor/login', status_code=status.HTTP_200_OK)
def doctor_login(request:LoginRequest):
 
    present = user_present(Queries.DOCTOR_TABLE_NAME.value,(request.email,request.password))
    
    if present:
        token = genToken(request.dict())
        data = {"message":"SUCCESS","result":[{"token":token}]}
        response = LoginResponse(**data)
        return response
    else:
        data = {"message":"User not found"}
        response = LoginResponse(**data)
        raise HTTPException(404,detail=[response.dict()])  
    

@app.post('/doctor/register',status_code=status.HTTP_201_CREATED)
def doctor_register(request:Doctor):
    present = user_present(Queries.DOCTOR_TABLE_NAME.value,(request.email,request.password))
    if not present:
        #register
        result = insert_operation(Queries.ADD_DOCTOR,request.tuple_for_insert())
        if result:
            return {"message":"success"}
        else:
            raise HTTPException(404,{"Some error occurred"})
    else:
        data = {"message":"User already exists"}
        raise HTTPException(404,data)


@app.post('/doctor/get-patients')
def doctor_get_patients(token:str):
    present,id = verifyToken(token)
    if present:
        result = fetch_operation(Queries.FETCH_PATIENTS_OF_DOCTOR,(id,))
        return {"message":"Success","result":result}
    #fetch operation
    else:
        data = {"message":"Invalid token","result":None}
        raise HTTPException(404,data)

@app.post('/doctor/all-records')
def doctor_all_records(token:str):
    present,id = verifyToken(token)
    if present:
        result = fetch_operation(Queries.MY_PATIENTS_ALL_RECORDS,(id,))
        #have to refine columns
        return {"message":"Success","result":result}
    #fetch operation
    else:
        data = {"message":"Invalid token","result":None}
        raise HTTPException(404,data)
    pass




#Patient Routes
@app.post('/patient/register')
def patient_register(request:Patient):
    present = user_present(Queries.PATIENT_TABLE_NAME.value,(request.email,request.password))
    if not present:
        #register
        result = insert_operation(Queries.ADD_PATIENT,request.tuple_for_insert())
        if result:
            return {"message":"success"}
        else:
            raise HTTPException(404,{"Some error occurred"})
    else:
        data = {"message":"User already exists"}
        raise HTTPException(404,data)

@app.post('/patient/login')
def patient_login(request:LoginRequest):
    present = user_present(Queries.PATIENT_TABLE_NAME.value,(request.email,request.password))
    
    if present:
        token = genToken(request.dict())
        data = {"message":"SUCCESS","result":[{"token":token}]}
        response = LoginResponse(**data)
        return response
    else:
        data = {"message":"User not found"}
        response = LoginResponse(**data)
        raise HTTPException(404,detail=[response.dict()])  

@app.post('/patient/predict')
async def predict_and_add_data(record:Record, background_tasks: BackgroundTasks):
    result = get_prediction(record)
    print("prediction done")
    background_tasks.add_task(insert_operation,Queries.INSERT_PATIENT_SINGLE_RECORD,result.tuple_for_insert())
    print("insert started")
    background_tasks.add_task(train_model)
    return {"message":"Success","result":[{"prediction":result.target}]}

@app.post('/patient/my-records')
def patient_my_records(token:str):
    present,id = verifyToken(token)
    if present:
        result = fetch_operation(Queries.FETCH_PATIENT_ALL_RECORDS,(id,))
        #have to refine columns
        return {"message":"Success","result":result}
    #fetch operation
    else:
        data = {"message":"Invalid token","result":None}
        raise HTTPException(404,data)
    pass

@app.get('/patient/record')#req params
def patient_date_records(request:TokenRequest):
    print(request.dict())
    return {"message":"suscces"}

#all patients


#DB routes

if __name__ == "__main__":
  uvicorn.run("main:app", reload=True)