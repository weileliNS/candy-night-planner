import {type State,LINES,UNIFORM,bit,cells,pop,won,unknown,posterior,probability,lineProbability,sample,rng,MAPS,transform} from './model';
export type Action={i:number;q:number;se?:number};
export type Result={mode:string;actions:Action[];value:number;ms:number;states:number;samples?:number;note?:string;tiedActions?:number[]};
export type Options={p:number;prior:number[];mode:string;exactMs:number;approxMs:number;rollouts?:number};
const key=(s:State)=>s.c+s.e*65536;
export function relevantMask(s:State){return LINES.filter(l=>!(l&s.e)).reduce((mask,l)=>mask|l,0)}
export function mandatoryMask(s:State){const lines=LINES.filter(l=>!(l&s.e));return lines.length?lines.reduce((mask,l)=>mask&l,65535)&unknown(s):0}
const entropy=(w:number[])=>-w.reduce((sum,v)=>sum+(v?v*Math.log2(v):0),0);
export function informationGain(s:State,p:number,prior:number[],i:number,w=posterior(s,p,prior)){
 const pc=probability(i,p,w);let gain=entropy(w);
 if(pc>0)gain-=pc*entropy(posterior({c:s.c|bit(i),e:s.e},p,prior));
 if(pc<1)gain-=(1-pc)*entropy(posterior({c:s.c,e:s.e|bit(i)},p,prior));
 return Math.max(0,gain);
}
function rankActions(actions:Action[],s:State,p:number,prior:number[],tolerance:number){
 const value=Math.min(...actions.map(a=>a.q)),w=posterior(s,p,prior);
 const tied=actions.filter(a=>a.q<=value+tolerance),rest=actions.filter(a=>a.q>value+tolerance);
 // Information is only a secondary preference inside a cost tie. Exact search
 // allows roundoff tolerance; rollout requires identical sample means.
 const gains=new Map(tied.map(a=>[a.i,informationGain(s,p,prior,a.i,w)]));
 tied.sort((a,b)=>Math.abs(gains.get(b.i)!-gains.get(a.i)!)>1e-12?gains.get(b.i)!-gains.get(a.i)!:a.i-b.i);
 rest.sort((a,b)=>a.q-b.q||a.i-b.i);
 return {actions:[...tied,...rest],value,tiedActions:tied.map(a=>a.i)};
}
export class Exact {
 cache=new Map<number,number>();nodes=0;deadline=Infinity;symmetric:boolean;
 constructor(public p:number,public prior=UNIFORM){this.symmetric=prior.every(x=>x===prior[0])}
 canonical(s:State){let k=key(s);if(this.symmetric)for(const m of MAPS.slice(1))k=Math.min(k,transform(s.c,m)+transform(s.e,m)*65536);return k}
 value(s:State):number{if(won(s))return 0;if((++this.nodes&127)===0&&(performance.now()>this.deadline||this.cache.size>350000))throw Error('Exact search reached its time or cache budget.');const k=this.canonical(s),v=this.cache.get(k);if(v!==undefined)return v;
 const mandatory=mandatoryMask(s);
 // These unknown cells occur on EVERY unblocked winning line, not merely on
 // every possible hidden line. Every successful policy must reveal all of them.
 // Their deterministic reveals commute with other moves, so factor their cost.
 if(mandatory){const v=pop(mandatory)+this.value({c:s.c|mandatory,e:s.e});this.cache.set(k,v);return v}
 const w=posterior(s,this.p,this.prior);let best=Infinity;
 // Outside this union, a cell cannot complete any line and is independent of H
 // given S. It provides neither progress nor useful information; omit it.
 const actions=cells(unknown(s)&relevantMask(s)).map(i=>({i,p:probability(i,this.p,w)})).sort((a,b)=>b.p-a.p);
 for(const a of actions){let q=1;if(a.p>0)q+=a.p*this.value({c:s.c|bit(a.i),e:s.e});if(a.p<1)q+=(1-a.p)*this.value({c:s.c,e:s.e|bit(a.i)});best=Math.min(best,q);if(best===1)break}
 this.cache.set(k,best);return best;
 }
 solve(s:State,ms:number):Result{const start=performance.now();this.deadline=start+ms;posterior(s,this.p,this.prior);if(won(s))return {mode:'Exact Optimal',actions:[],value:0,ms:performance.now()-start,states:this.nodes};
 const w=posterior(s,this.p,this.prior),relevant=relevantMask(s),value=this.value(s),actions=cells(unknown(s)).map(i=>{if(!(relevant&bit(i)))return {i,q:1+value};const p=probability(i,this.p,w);return {i,q:1+(p>0?p*this.value({c:s.c|bit(i),e:s.e}):0)+(p<1?(1-p)*this.value({c:s.c,e:s.e|bit(i)}):0)}});
 return {mode:'Exact Optimal',...rankActions(actions,s,this.p,this.prior,1e-10),ms:performance.now()-start,states:this.nodes};}
}
export function heuristic(s:State,p:number,prior:number[],kind:string,rand:()=>number):number{
 const u=cells(unknown(s)),w=posterior(s,p,prior);if(kind==='random')return u[Math.floor(rand()*u.length)];
 let best=-Infinity,choice=u[0];
 for(const i of u){const pc=probability(i,p,w);let score=pc;
 if(kind==='greedy'||kind==='rollout'){score=-Infinity;LINES.forEach(l=>{if(!(l&s.e)&&(l&bit(i))){const n=pop(l&unknown(s)),chance=lineProbability(l,s,p,w);const v=kind==='greedy'?pop(l&s.c)*10+pc:chance/n+.025*pc;score=Math.max(score,v)}})}
 if(kind==='info')score=informationGain(s,p,prior,i,w);
 if(score>best+1e-12){best=score;choice=i}}
 return choice;
}
export function approximate(s:State,o:Options):Result{
 const start=performance.now(),w=posterior(s,o.p,o.prior),u=cells(unknown(s));if(won(s))return {mode:'Exact Optimal',actions:[],value:0,ms:0,states:0};
 const sums=u.map(()=>0),squares=u.map(()=>0),rand=rng(91827+key(s));let n=0;
 // Paired posterior worlds across actions. Each Q estimates a legal, adaptive
 // completion policy, not the Bellman optimum. Full rounds avoid unequal sample counts.
 do{const board=sample(s,o.p,w,rand);u.forEach((i,j)=>{let t={...s},next=i,cost=0;while(!won(t)){if(board&bit(next))t.c|=bit(next);else t.e|=bit(next);cost++;if(!won(t))next=heuristic(t,o.p,o.prior,'rollout',rand)}sums[j]+=cost;squares[j]+=cost*cost});n++}while(n<(o.rollouts??100000)&&performance.now()-start<o.approxMs);
 const actions=u.map((i,j)=>({i,q:sums[j]/n,se:n>1?Math.sqrt(Math.max(0,(squares[j]-sums[j]**2/n)/(n-1))/n):undefined}));
 return {mode:'Approximate',...rankActions(actions,s,o.p,o.prior,0),ms:performance.now()-start,states:0,samples:n,note:'Posterior sampling with adaptive rollouts. Q estimates a continuation policy. The 95% interval covers sampling error for one action only, not selection bias or planning approximation error.'};
}
let retained:Exact|undefined,signature='';
export function solve(s:State,o:Options):Result{const start=performance.now();posterior(s,o.p,o.prior);if(o.mode==='heuristic'){const i=heuristic(s,o.p,o.prior,'probability',rng(1));return won(s)?{mode:'Exact Optimal',value:0,actions:[],ms:0,states:0}:{mode:'Heuristic',value:NaN,actions:[{i,q:NaN}],ms:performance.now()-start,states:0,note:'Ranked by candy probability only; clearing cost is not estimated.'}}
 if(o.mode!=='approx'){const sig=JSON.stringify([o.p,o.prior]);if(sig!==signature||!retained){retained=new Exact(o.p,o.prior);signature=sig}try{return retained.solve(s,o.exactMs)}catch(e){if(o.mode==='exact')throw e;const result=approximate(s,o);result.note='Exact search reached its budget; switched to approximate planning. '+result.note;result.ms=performance.now()-start;return result}}
 return approximate(s,o);
}
