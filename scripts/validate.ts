import {writeFileSync,mkdirSync} from 'node:fs';
import {simulate} from '../src/simulation';
import {type Stats} from '../src/simulation';
import {SAMPLES,parse,LINES,NAMES,pop,bit} from '../src/model';
const rows:Stats[]=[];const count=Number(process.argv[2]??1000),budget=Number(process.argv[3]??1000);
simulate(count,budget,row=>{rows.push(row);console.log(`p=${row.p.toFixed(4)} ${row.strategy}: n=${row.n}, mean=${row.mean.toFixed(3)}, ${row.status}`)});
mkdirSync('validation',{recursive:true});writeFileSync('validation/simulation.json',JSON.stringify({seed:20261028,count,budgetMsPerStrategy:budget,generatedAt:new Date().toISOString(),rows},null,2));
const samples=SAMPLES.map((text,i)=>{const s=parse(text);return {name:'ABC'[i],text,candy:pop(s.c),empty:pop(s.e),possibleLines:LINES.flatMap((l,j)=>!(l&s.e)?[NAMES[j]]:[]),completeIfUIsCandy:LINES.flatMap((l,j)=>((s.c|Array.from({length:16},(_,k)=>bit(k)).filter(b=>!((s.c|s.e)&b)).reduce((a,b)=>a|b,0))&l)===l?[NAMES[j]]:[])}});writeFileSync('validation/samples.json',JSON.stringify(samples,null,2));
