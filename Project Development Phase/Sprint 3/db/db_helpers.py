import psycopg2
import urllib.parse as up
from .db_enum import Queries
import json
import psycopg2.extras

#need to put it in an environment variable
DATABASE_URL = "url"
up.uses_netloc.append("postgres")
url = up.urlparse(DATABASE_URL)


QUERY_MAP={
    ""
}

def get_connection():
    conn = psycopg2.connect(database=url.path[1:],user=url.username,password=url.password,host=url.hostname,port=url.port)
    return conn

def close_connection(conn):
    conn.close()
    pass


def insert_operation(query,args):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(query.value,args)
        conn.commit()
        res = True
    except Exception as ex:
        print(ex)
        res = False
    cur.close()
    close_connection(conn)
    return res
    

def fetch_operation(query,args):
    conn = get_connection()
    cur = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    try:
        cur.execute(query.value,args)
        result = (cur.fetchall())

    except Exception as ex:
        print(ex)
        result = {}
    conn.commit()
    cur.close()
    return result

def create_table(query):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query.value)
    conn.commit()
    cur.close()
    close_connection(conn)


def create_all_tables():
    queries = [Queries.CREATE_DOCTOR_TABLE,Queries.CREATE_PATIENT_TABLE,Queries.CREATE_RECORD_TABLE]
    conn = get_connection()
    cur = conn.cursor()
    for query in queries:
        cur.execute(query.value)
        conn.commit()
    cur.close()
    close_connection(conn)

def delete_all_tables():
    table_names = [Queries.RECORD_TABLE_NAME,Queries.PATIENT_TABLE_NAME,Queries.DOCTOR_TABLE_NAME]
    conn = get_connection()
    cur = conn.cursor()
    for table in table_names:
        cur.execute('DROP TABLE '+table.value)
        conn.commit()
    cur.close()
    close_connection(conn)

def user_present(user,args):
    conn = get_connection()
    cur = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    query = "SELECT COUNT(id) FROM "+user+" WHERE email='"+args[0] + "' and password='" + args[1]+"'"
    cur.execute(query)
    result = (cur.fetchall())
    conn.commit()
    cur.close()
    
    if result and len(result)>0 and 'count' in result[0]:
        #print(result[0]["count"])
        return result[0]["count"]==1
    return False
        
def get_id_of_user(user,args):
    conn = get_connection()
    cur = conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    query = "SELECT id FROM "+user+" WHERE email='"+args[0] + "' and password='" + args[1]+"'"
    cur.execute(query)
    result = (cur.fetchall())
    conn.commit()
    cur.close()
    
    if result and len(result)>0 and 'id' in result[0]:
        #print(result[0]["count"])
        return result[0]["id"]
    return -1
