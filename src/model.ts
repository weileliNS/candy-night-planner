export type State = { c:number; e:number };
export const FULL=65535;
export const bit=(i:number)=>1<<i;
export const pop=(x:number)=>{let n=0;for(;x;x&=x-1)n++;return n};
export const cells=(m:number)=>Array.from({length:16},(_,i)=>i).filter(i=>m&bit(i));
export const LINES=[...Array.from({length:4},(_,r)=>15<<(r*4)),...Array.from({length:4},(_,c)=>0x1111<<c),0x8421,0x1248];
export const NAMES=['Row 1','Row 2','Row 3','Row 4','Column 1','Column 2','Column 3','Column 4','Main diagonal','Anti-diagonal'];
export const won=(s:State)=>LINES.some(l=>(s.c&l)===l);
export const unknown=(s:State)=>FULL&~(s.c|s.e);
export const UNIFORM=Array(10).fill(.1) as number[];
export function posterior(s:State,p:number,prior=UNIFORM):number[]{
 if(!Number.isFinite(p)||p<0||p>1||!Number.isInteger(s.c)||!Number.isInteger(s.e)||s.c<0||s.e<0||s.c>FULL||s.e>FULL||(s.c&s.e))throw Error('Invalid probability or board masks.');
 if(prior.length!==10||prior.some(x=>!Number.isFinite(x)||x<0)||!prior.some(x=>x>0))throw Error('The prior needs 10 nonnegative weights with a positive sum.');
 // P(S|h)=0 if h intersects E; otherwise p^(|C|-|C∩h|)(1-p)^|E|.
 // The common factor cancels: posterior ∝ prior(h) p^(-|C∩h|).
 // At p=0 use support C⊆h; at p=1 only E=∅ is possible. No 0^0 or infinities.
 const logs=LINES.map((l,i)=>{if(l&s.e||prior[i]===0||p===1&&s.e||p===0&&(s.c&~l))return -Infinity;return Math.log(prior[i])-(p>0&&p<1?pop(s.c&l)*Math.log(p):0)});
 const max=Math.max(...logs);if(max===-Infinity)throw Error('This board is incompatible with the model: no hidden line is possible.');
 const weights=logs.map(x=>Math.exp(x-max)),sum=weights.reduce((a,b)=>a+b,0);return weights.map(x=>x/sum);
}
export function probability(i:number,p:number,w:number[]){const inside=w.some((v,h)=>v>0&&!!(LINES[h]&bit(i))),outside=w.some((v,h)=>v>0&&!(LINES[h]&bit(i)));if(!inside)return p;if(!outside)return 1;return Math.max(0,Math.min(1,p+(1-p)*w.reduce((a,v,h)=>a+((LINES[h]&bit(i))?v:0),0)))}
export function lineProbability(l:number,s:State,p:number,w:number[]){if(l&s.e)return 0;const u=l&unknown(s);return w.reduce((a,v,h)=>a+v*p**pop(u&~LINES[h]),0)}
export function rng(seed:number){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296}}
export function sample(s:State,p:number,w:number[],rand:()=>number){let z=rand(),h=9;for(let j=0;j<10;j++){z-=w[j];if(z<=0){h=j;break}}let board=s.c|LINES[h];for(const i of cells(unknown(s)&~board))if(rand()<p)board|=bit(i);return board}
export function parse(text:string):State{const a=text.toUpperCase().replace(/[^CEU]/g,'');if(a.length!==16)throw Error('Enter 16 C / E / U symbols in row order.');let c=0,e=0;[...a].forEach((v,i)=>{if(v==='C')c|=bit(i);if(v==='E')e|=bit(i)});return {c,e}}
export const format=(s:State)=>Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>s.c&bit(r*4+c)?'C':s.e&bit(r*4+c)?'E':'U').join(' ')).join('\n');
export const SAMPLES=['E E C C\nE C C E\nC C C E\nE C U C','C C E E\nC C C U\nC C E E\nE E C C','C C C E\nC C E E\nE E U C\nE E C C'];
export const MAPS=Array.from({length:8},(_,t)=>Array.from({length:16},(_,i)=>{let r=i>>2,c=i%4;if(t>=4)c=3-c;for(let k=0;k<t%4;k++)[r,c]=[c,3-r];return r*4+c}));
export const transform=(m:number,map:number[])=>cells(m).reduce((a,i)=>a|bit(map[i]),0);
