from enum import Enum
class Queries(Enum):
    DOCTOR_TABLE_NAME = "doctor"
    PATIENT_TABLE_NAME = "patient"
    RECORD_TABLE_NAME = "records"

    CREATE_DOCTOR_TABLE = "CREATE TABLE doctor (id serial primary key,name text,email text,password text,dept text)"
    CREATE_PATIENT_TABLE = "CREATE TABLE patient (id serial primary key,name text,email text,password text,doc_id integer,FOREIGN KEY (doc_id) REFERENCES doctor(id) )"
    CREATE_RECORD_TABLE = "CREATE TABLE records (id integer,age integer,sex integer,cp integer,trestbps integer,chol integer,fbs integer,restecg integer,thalach integer,exang integer,oldpeak numeric, \
                            slope integer,ca integer,thal integer,target integer,date text,time text,FOREIGN KEY (id) REFERENCES patient(id) )"
    
    ADD_DOCTOR = "INSERT INTO doctor(name,email,password,dept) VALUES (%s,%s,%s,%s)"
    ADD_PATIENT = "INSERT INTO patient(name,email,password,doc_id) VALUES (%s,%s,%s,%s)"

    FETCH_DOCTORS = "SELECT * FROM doctor"
    FETCH_PATIENTS = "SELECT * FROM patients"
    FETCH_ALL_PATIENTS_ID = ""
    FETCH_ALL_PATIENTS_NAME_ID = "SELECT id,name FROM patient"

    INSERT_PATIENT_SINGLE_RECORD = "INSERT INTO records (id,age ,sex ,cp ,trestbps ,chol ,fbs ,restecg ,thalach , exang ,oldpeak , \
                            slope ,ca ,thal ,target ,date ,time ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    FETCH_PATIENT_SINGLE_RECORD = ""
    FETCH_PATIENT_ALL_RECORDS = "SELECT * FROM records WHERE id=%s"
    FETCH_PATIENTS_OF_DOCTOR = "SELECT id,name,email FROM patient WHERE doc_id=%s"
    MY_PATIENTS_ALL_RECORDS = "SELECT * FROM records LEFT JOIN patient ON records.id=patient.id AND patient.doc_id=%s"
    FETCH_ALL_RECORDS = "SELECT * FROM records"

    DOCTOR_LOGIN = "SELECT COUNT(id) FROM doctor WHERE email=%s AND password=%s"
    PATIENT_LOGIN = "SELECT COUNT(id) FROM patient WHERE email=%s AND password=%s"