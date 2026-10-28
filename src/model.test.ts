import {describe,it,expect} from 'vitest';
import {type State,LINES,bit,cells,pop,posterior,probability,lineProbability,UNIFORM,won,unknown,MAPS,transform,parse,SAMPLES,sample,rng} from './model';
import {Exact,approximate,solve} from './solver';
function brute(s:State,p:number):number{if(won(s))return 0;const w=posterior(s,p);return Math.min(...cells(unknown(s)).map(i=>{const q=probability(i,p,w);return 1+(q>0?q*brute({c:s.c|bit(i),e:s.e},p):0)+(q<1?(1-q)*brute({c:s.c,e:s.e|bit(i)},p):0)}))}
describe('规则与贝叶斯模型',()=>{
 it('恰好10条唯一4格合法线',()=>{expect(new Set(LINES).size).toBe(10);expect(LINES.every(l=>pop(l)===4)).toBe(true);expect(LINES).toContain(15);expect(LINES).toContain(0x1111);expect(LINES).toContain(0x8421);expect(LINES).toContain(0x1248)});
 it('所有16格覆盖数',()=>expect(Array.from({length:16},(_,i)=>LINES.filter(l=>l&bit(i)).length)).toEqual([3,2,2,3,2,3,3,2,2,3,3,2,3,2,2,3]));
 it('初始后验均匀',()=>expect(posterior({c:0,e:0},.4444)).toEqual(UNIFORM));
 it('空格排除对应保底线',()=>{const w=posterior({c:0,e:1},.4);LINES.forEach((l,i)=>expect(w[i]).toBeCloseTo(l&1?0:1/7))});
 it('糖果似然比为1/p',()=>{const w=posterior({c:1,e:0},.4);expect(w[0]/w[1]).toBeCloseTo(2.5)});
 it('未知格概率',()=>expect(probability(0,.4,UNIFORM)).toBeCloseTo(.58));
 it('区分后验与整线概率',()=>{const w=posterior({c:0,e:0},.4);expect(lineProbability(15,{c:0,e:0},.4,w)).toBeGreaterThan(w[0])});
 it('对全65536棋盘的生成分布枚举核对后验预测',()=>{const p=.4,s={c:bit(0)|bit(5),e:bit(2)},i=7;let mass=0,candyMass=0;const hMass=Array(10).fill(0);for(let b=0;b<=65535;b++){if((b&s.c)!==s.c||(b&s.e))continue;LINES.forEach((l,h)=>{if((b&l)!==l)return;const bg=pop(b&~l),v=.1*p**bg*(1-p)**(12-bg);mass+=v;hMass[h]+=v;if(b&bit(i))candyMass+=v})}const w=posterior(s,p);w.forEach((v,h)=>expect(v).toBeCloseTo(hMass[h]/mass,10));expect(probability(i,p,w)).toBeCloseTo(candyMass/mass,10)});
 it('矛盾输入及非法参数被拒绝',()=>{expect(()=>posterior({c:1,e:1},.4)).toThrow();expect(()=>posterior({c:0,e:65535},.4)).toThrow();expect(()=>posterior({c:0,e:0},NaN)).toThrow();expect(()=>posterior({c:0,e:0},.4,Array(10).fill(0))).toThrow()});
 it('p=0只接受位于某条保底线内的糖果',()=>{expect(posterior({c:15,e:0},0)[0]).toBe(1);expect(()=>posterior({c:31,e:0},0)).toThrow()});
 it('p=1允许所有糖果并拒绝任何空格',()=>{expect(probability(5,1,posterior({c:0,e:0},1))).toBe(1);expect(()=>posterior({c:0,e:1},1)).toThrow()});
 it('真实样本保持原样：A列3、B行2、C主对角线',()=>{const possible=(s:State)=>LINES.map((l,i)=>!(l&s.e)?i:-1).filter(i=>i>=0);expect(possible(parse(SAMPLES[0]))).toEqual([6]);expect(possible(parse(SAMPLES[1]))).toEqual([1]);expect(possible(parse(SAMPLES[2]))).toEqual([8]);expect(posterior(parse(SAMPLES[2]),.4)[8]).toBe(1)});
 it('原矩阵背景统计应为17糖果19空格，U计为真实糖果',()=>{const states=SAMPLES.map(parse);expect(states.reduce((sum,s)=>sum+pop(s.c)+pop(unknown(s))-4,0)).toBe(17);expect(states.reduce((sum,s)=>sum+pop(s.e),0)).toBe(19)});
 it('后验抽样始终满足观测与保底',()=>{const s={c:1,e:bit(2)},p=.4,w=posterior(s,p),r=rng(1);for(let n=0;n<100;n++){const b=sample(s,p,w,r);expect((b&s.c)===s.c&&!(b&s.e)&&won({c:b,e:0})).toBe(true)}});
});
describe('Bellman 与规划',()=>{
 it('每条合法线都能使V=0，包括同时完成多条线',()=>{for(const l of [...LINES,65535])expect(new Exact(.4).solve({c:l,e:0},1000).value).toBe(0)});
 it('推断必胜仍需翻开：三个样本剩余值均为1',()=>{for(const i of [0,1,2])expect(new Exact(.4).solve(parse(SAMPLES[i]),1000).value).toBe(1)});
 it('p=1初始最少4次，p=0指定保底线也需4次',()=>{expect(new Exact(1).solve({c:0,e:0},5000).value).toBe(4);const prior=[1,...Array(9).fill(0)];expect(new Exact(0,prior).solve({c:0,e:0},1000).value).toBe(4)});
 it('近终局精确结果等于无缓存暴力枚举',()=>{const states=[{c:bit(0)|bit(1)|bit(4)|bit(5)|bit(8)|bit(9)|bit(10)|bit(15),e:bit(3)|bit(6)|bit(7)|bit(11)},parse(SAMPLES[0]),parse(SAMPLES[1])];for(const s of states)expect(new Exact(.37).solve(s,5000).value).toBeCloseTo(brute(s,.37),10)});
 it('D4旋转镜像不改变值',()=>{const s={c:bit(0)|bit(1)|bit(4)|bit(5)|bit(8)|bit(9)|bit(10)|bit(15),e:bit(3)|bit(6)|bit(7)|bit(11)},ex=new Exact(.4),v=ex.solve(s,5000).value;for(const map of MAPS)expect(ex.solve({c:transform(s.c,map),e:transform(s.e,map)},1000).value).toBeCloseTo(v,10)});
 it('非均匀先验不做无效的对称合并',()=>expect(new Exact(.4,[1,...Array(9).fill(0)]).symmetric).toBe(false));
 it('近似估计确定必胜格为1并诚实标识',()=>{const r=approximate(parse(SAMPLES[0]),{p:.4,prior:UNIFORM,mode:'approx',exactMs:0,approxMs:1000,rollouts:20});expect(r.mode).toBe('Approximate');expect(r.value).toBe(1);expect(r.actions[0].se).toBe(0);expect(r.samples).toBe(20)});
 it('启发式不伪造期望成本',()=>{const r=solve({c:0,e:0},{p:.4,prior:UNIFORM,mode:'heuristic',exactMs:1,approxMs:1});expect(r.mode).toBe('Heuristic');expect(r.value).toBeNaN()});
});
