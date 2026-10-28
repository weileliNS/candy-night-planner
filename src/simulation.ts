import {type State,UNIFORM,rng,sample,won,bit} from './model';
import {Exact,heuristic,approximate,type Options} from './solver';
export const STRATEGIES=['Random','Max Candy Probability','Max Information Gain','Line Completion Greedy','Exact Bellman','Approximate Planner'];
export const PS=[.2,.3,16/36,.5,.6,.8];
export type Stats={p:number;strategy:string;n:number;requested:number;mean:number;median:number;p90:number;p95:number;worst:number;tokens:number;ms:number;saving:number;pairedRandom:number;status:string};
export function simulate(count:number,budgetMs:number,emit:(row:Stats)=>void,ps=PS){
 for(const p of ps){const rand=rng(20261028),boards=Array.from({length:count},()=>sample({c:0,e:0},p,UNIFORM,rand));let baseline:number[]=[];
 for(let strategy=0;strategy<6;strategy++){
 const start=performance.now(),values:number[]=[],r=rng(42),ex=new Exact(p);let status='Complete';
 for(let j=0;j<count;j++){if(performance.now()-start>=budgetMs){status='Time budget reached; compare shared board prefixes only';break}const s:State={c:0,e:0};let cost=0;
 try{while(!won(s)){let i:number;if(strategy===4){i=ex.solve(s,Math.max(1,Math.min(1000,budgetMs-(performance.now()-start)))).actions[0].i}else if(strategy===5){const o:Options={p,prior:UNIFORM,mode:'approx',exactMs:0,approxMs:Math.max(1,Math.min(8,budgetMs-(performance.now()-start))),rollouts:24};i=approximate(s,o).actions[0].i}else{i=heuristic(s,p,UNIFORM,['random','probability','info','greedy'][strategy],r)}if(boards[j]&bit(i))s.c|=bit(i);else s.e|=bit(i);cost++;if(cost>16)throw Error('Strategy failed to terminate')}values.push(cost)}catch{status='Exact budget reached; no approximate substitution';break}}
 if(strategy===0)baseline=values.slice();const n=values.length,sorted=[...values].sort((a,b)=>a-b),mean=n?values.reduce((a,b)=>a+b,0)/n:NaN,pairedN=Math.min(n,baseline.length),baseMean=pairedN?baseline.slice(0,pairedN).reduce((a,b)=>a+b,0)/pairedN:NaN,ownMean=pairedN?values.slice(0,pairedN).reduce((a,b)=>a+b,0)/pairedN:NaN;
 emit({p,strategy:STRATEGIES[strategy],n,requested:count,mean,median:n?(sorted[Math.floor((n-1)/2)]+sorted[Math.ceil((n-1)/2)])/2:NaN,p90:sorted[Math.ceil(.9*n)-1]??NaN,p95:sorted[Math.ceil(.95*n)-1]??NaN,worst:sorted[n-1]??NaN,tokens:mean*5,ms:performance.now()-start,saving:baseMean-ownMean,pairedRandom:pairedN,status});
 }
 }
}
