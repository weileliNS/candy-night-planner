import {solve} from './solver';
import {simulate} from './simulation';
self.onmessage=e=>{const {id,type,...data}=e.data;try{if(type==='simulation'){simulate(data.count,data.budget,(row)=>self.postMessage({id,row}));self.postMessage({id,done:true})}else self.postMessage({id,result:solve(data.state,data.options)})}catch(error){self.postMessage({id,error:error instanceof Error?error.message:String(error)})}};
