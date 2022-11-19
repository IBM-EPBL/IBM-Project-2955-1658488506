import React,{useEffect,useState} from 'react'
import NavBar from '../components/navbar/NavBar'
import Header from '../components/header/Header'
import { Table } from '../components/tables/Table'
import { BarChart } from '../components/charts/BarChart'
import { SelectionForms } from '../components/forms/SelectionForms'
import { LineChart } from '../components/charts/LineChart'
import { useCookies } from 'react-cookie';


export const PatientProgress = () => {
  const [patients,setPatients] = useState([]);
  const [cookies, setCookie, removeCookie] = useCookies();
  const [patient,setPatient] = useState([])
  const token = cookies.token
  const [selectedX,setX] = useState([])
  const [selectedY,setY] = useState([])
  const [allrecords,setAllrecords] = useState([])
  const [filteredRecords,setFilteredRecords] = useState([])
  const [labelsX,setLabelsX] = useState([])
  const [dataY,setDataY] = useState([])
  const X_LIST=["slope","trestbps","chol","thalach","target"]
const Y_LIST=["ca","target","trestbps","chol","thalach"]

  useEffect(() => {
    fetch('http://127.0.0.1:8000/doctor/get-patients',{
                method: 'POST',
             body: JSON.stringify({
                token:token
             }),
             headers: {
                'Content-type': 'application/json; charset=UTF-8',
             },
          })
             .then((res) => res.json())
             .then((res) => {
              console.log(res)
              setPatients(res.result)
             })
             .catch((err) => {
    			console.log(err)
                console.log(err.message);
             });
    },[])

useEffect(()=>{
  // console.log(patient)
  var newArray = allrecords.filter(function(el){return patient.id==  el.id})
  console.log(newArray)
  setFilteredRecords(newArray)
},([patient]))

useEffect(()=>{
  console.log("useEffect Xaxis")
 var temp=[]
 for (let obj in filteredRecords){
  console.log(filteredRecords[obj][selectedX])
  temp.push(filteredRecords[obj][selectedX])
 } 
 console.log(temp)
 setLabelsX(temp)
},([selectedX]))

useEffect(()=>{
  console.log("useEffect Yaxis")
  var temp=[]
  for (let obj in filteredRecords){
    temp.push(filteredRecords[obj][selectedY])
  } 
  console.log(temp)
  setDataY(temp)
 },([selectedY]))

    useEffect(() => {
      fetch('http://127.0.0.1:8000/doctor/all-records',{
                  method: 'POST',
               body: JSON.stringify({
                  token:token
               }),
               headers: {
                  'Content-type': 'application/json; charset=UTF-8',
               },
            })
               .then((res) => res.json())
               .then((res) => {
                console.log(res)
                setAllrecords(res.result)
               })
               .catch((err) => {
            console.log(err)
                  console.log(err.message);
               });
      },[])
  

  return (
    <div className="App overflow-y-hidden ">
     <Header/>
     <div className='w-full min-h-[90vh] grid grid-cols-12'>
     <NavBar/>
     <div className='grid grid-cols-1 col-span-7 m-5 p-5 w-full'>
     <SelectionForms patients={patients} listX={X_LIST} listY={Y_LIST} selectedX={selectedX} selectedY={selectedY} setPatient={setPatient} setX={setX} setY={setY} />  
      <BarChart  labels={labelsX} data={dataY}/>

     </div>
     </div>
    </div>
  )
}