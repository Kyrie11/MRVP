from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
from .schema import ACTION_NAMES,ACTION_VECTORS,BOTTLE_NECKS,CTX_DIM,DEG_DIM,HIST_LEN,MAX_AGENTS,ACTOR_FEAT_DIM,MECH_DIM,NUM_ACTIONS,STATE_DIM,SIDE_NAMES,RECOVERY_HORIZON,harm_bin_from_rho
FAMILIES=['rear_end_blocked_forward_corridor','side_swipe_near_boundary','oblique_intersection_impact','cut_in_unavoidable_contact','boundary_critical_non_contact','low_friction_recovery','actuator_degradation_after_impact','dense_agent_secondary_exposure']

def _split_for_root(i:int,n:int)->str:
    f=i/max(1,n); return 'train' if f<0.70 else ('val' if f<0.82 else ('cal' if f<0.90 else 'test'))

def annotate_root_counterfactual_fields(rows: List[Dict], expected_actions:int=NUM_ACTIONS)->None:
    if not rows: return
    min_bin=min(int(r.get('harm_bin',999)) for r in rows); flags=[int(r.get('harm_bin',999))==min_bin for r in rows]
    ss=[float(r.get('s_star',0.0)) for r,f in zip(rows,flags) if f]; spread=float(max(ss)-min(ss)) if len(ss)>=2 else 0.0
    for r,f in zip(rows,flags):
        r.update({'root_action_count':len(rows),'root_expected_action_count':expected_actions,'root_action_complete':len(rows)==expected_actions,'root_min_harm_bin':min_bin,'is_harm_admissible':bool(f),'root_admissible_count':int(sum(flags)),'root_same_harm_s_spread':spread,'has_event_tokens':0})

def make_synthetic_rows(n_roots:int=240, actions_per_root:int=NUM_ACTIONS, seed:int=7, harm_thresholds:Sequence[float]=(0.5,2.0,4.0,7.0,11.0))->List[Dict]:
    rng=np.random.default_rng(seed); rows=[]
    for r in range(n_roots):
        family_id=r%len(FAMILIES); family=FAMILIES[family_id]; split=_split_for_root(r,n_roots); root_id=f'syn_{r:06d}'
        base_speed=float(rng.uniform(5,18)); friction=float(rng.choice([0.95,0.75,0.35],p=[0.65,0.25,0.10]));
        if family_id==5: friction=float(rng.uniform(0.25,0.55))
        actor_density=float(rng.uniform(0.1,1.0) if family_id in (0,2,7) else rng.uniform(0.0,0.5)); lane_width=float(rng.uniform(2.8,4.2)); boundary_side=int(rng.choice([-1,1])); damage_class=int(rng.integers(0,4) if family_id in (0,2,6) else rng.integers(0,2))
        base_x=float(rng.normal(0,4)); base_y=float(rng.normal(0,2)); base_yaw=float(rng.normal(0,0.15)); route_dx=float(rng.uniform(20,60)); route_dy=float(rng.normal(0,3)); town=f'Town{int(rng.integers(1,8)):02d}'
        root_rows=[]
        for a in range(actions_per_root):
            action_name=ACTION_NAMES[a%len(ACTION_NAMES)]; action_vec=ACTION_VECTORS[a%len(ACTION_VECTORS)].astype(float); steer, throttle, brake, duration=action_vec
            steer_effect=float(steer)*rng.uniform(0.6,1.4); brake_effect=float(brake)*rng.uniform(0.7,1.3); contact=family_id!=4
            harm_base=max(0.0, base_speed*(0.18+0.04*family_id)+rng.normal(0,0.25)); harm_action=harm_base-0.35*brake_effect+0.12*abs(steer_effect)+rng.normal(0,0.18)
            if not contact: harm_action=rng.uniform(0.0,0.4)
            rho_imp=float(max(0.0,harm_action)); harm_bin=harm_bin_from_rho(rho_imp,harm_thresholds)
            vx0=base_speed*np.cos(base_yaw); vy0=base_speed*np.sin(base_yaw); yaw_rate0=float(rng.normal(0,0.05)); beta0=float(rng.normal(0,0.03)); x_minus=np.zeros(STATE_DIM,dtype=np.float32); x_minus[:]=[base_x,base_y,base_yaw,vx0,vy0,yaw_rate0,0,0,beta0,steer,brake,throttle]
            yaw_reset=(0.15+0.05*family_id)*steer_effect+(0.08*damage_class)*rng.normal(); speed_drop=(2*brake_effect+0.25*rho_imp+rng.normal(0,0.25)) if contact else brake_effect; lat_shift=0.4*steer_effect+rng.normal(0,0.15)
            x_plus=x_minus.copy(); new_speed=max(0.15,base_speed-speed_drop); new_yaw=base_yaw+yaw_reset; beta_plus=float(beta0+np.arctan2(steer_effect*0.8,max(new_speed,1e-3))); vx1=new_speed*np.cos(new_yaw+beta_plus); vy1=new_speed*np.sin(new_yaw+beta_plus)
            x_plus[0]+=duration*base_speed*np.cos(base_yaw)*0.65; x_plus[1]+=lat_shift; x_plus[2]=new_yaw; x_plus[3]=vx1; x_plus[4]=vy1; x_plus[5]=yaw_rate0+yaw_reset/max(duration,0.05); x_plus[6]=(vx1-vx0)/max(duration,0.05); x_plus[7]=(vy1-vy0)/max(duration,0.05); x_plus[8]=beta_plus; x_plus[9]=steer; x_plus[10]=brake; x_plus[11]=throttle
            steering_scale=max(0.25,1-0.12*damage_class-rng.uniform(0,0.08)); brake_scale=max(0.25,1-0.10*damage_class-rng.uniform(0,0.08)); throttle_scale=max(0.25,1-0.07*damage_class); delay=rng.uniform(0.02,0.18)+0.03*damage_class; d_deg=np.asarray([steering_scale,brake_scale,throttle_scale,delay,friction,damage_class/3.0],dtype=np.float32)
            clearance=lane_width/2-abs(base_y+lat_shift)-0.9; return_length=max(0.0,route_dx-3*actor_density-2*abs(lat_shift)); secondary_clear=max(0.0,12-7*actor_density-0.4*rho_imp+rng.normal(0,1.0))
            r_sec=secondary_clear-2.0-0.4*actor_density-0.2*abs(yaw_reset)+rng.normal(0,0.3); r_road=clearance+0.7-0.2*abs(steer_effect)+rng.normal(0,0.15); r_stab=1.6*friction-abs(x_plus[5])*0.55-abs(x_plus[8])*0.35-0.1*base_speed/10+rng.normal(0,0.15); r_ctrl=min(steering_scale,brake_scale)-0.35-0.25*abs(steer)-0.2*delay+rng.normal(0,0.08); r_return=0.08*return_length+0.2*r_road+0.15*r_sec-1.2*actor_density+rng.normal(0,0.25)
            if family_id==0: r_sec-=1.2*actor_density; r_return-=0.8
            elif family_id==1: r_road-=0.7
            elif family_id==5: r_stab-=0.9
            elif family_id==6: r_ctrl-=0.7
            r_star=np.asarray([r_sec,r_road,r_stab,r_ctrl,r_return],dtype=np.float32)
            event_type='boundary' if family_id==4 else ('stability' if family_id==5 else ('control' if family_id==6 else ('contact' if contact else 'none')))
            h_ctx=np.zeros(CTX_DIM,dtype=np.float32); h_ctx[:16]=[0,-base_yaw,lane_width,lane_width/2+base_y,lane_width/2-base_y,secondary_clear,8,secondary_clear,max(0.1,secondary_clear/max(base_speed,0.1)),return_length,actor_density,13.9,friction,rng.uniform(0,1),0,lane_width]; h_ctx[16:24]=[0,-base_y,route_dx,route_dy,0,actor_density*0.2,1,boundary_side]; h_ctx[24:29]=[int(town[-2:])/10,family_id/max(1,len(FAMILIES)-1),1.0 if contact else 0.0,damage_class/3,0.5]
            o_hist=np.zeros((HIST_LEN,MAX_AGENTS,ACTOR_FEAT_DIM),dtype=np.float32)
            for t in range(HIST_LEN):
                dt=(t-HIST_LEN+1)*0.1; o_hist[t,0,:]=[base_x+base_speed*dt,base_y,base_yaw,base_speed,0,4.6,1.9,1,1]
                for ag in range(1,min(MAX_AGENTS,1+int(2+actor_density*8))): o_hist[t,ag,:]=[base_x+rng.uniform(6,35)+base_speed*dt*rng.uniform(0.5,1.1),base_y+rng.normal(0,3.2),rng.normal(0,0.2),base_speed*rng.uniform(0.3,1.2),rng.normal(0,0.5),4.5,1.8,1,1]
            z=np.zeros(MECH_DIM,dtype=np.float32); side_id=3 if steer>0.2 else (2 if steer<-0.2 else int(rng.choice([0,1,4]))); z[0]=1.0 if contact else 0.0; z[1+side_id]=1.0; normal=np.asarray([-np.sign(steer) if abs(steer)>0.1 else 1.0,-boundary_side],dtype=np.float32); normal/=np.linalg.norm(normal)+1e-6; z[6:8]=normal; z[8]=clearance if not contact else -0.1*rho_imp; z[10]=rho_imp; reset=x_plus-x_minus; z[11:18]=[reset[3],reset[4],np.hypot(x_plus[3],x_plus[4])-base_speed,reset[2],reset[5],reset[8],reset[1]]; z[18:24]=[lane_width,clearance,return_length,secondary_clear,np.sign(route_dy),1]; z[24:29]=[steering_scale,brake_scale,delay,friction,damage_class/3]; z[29:32]=[rng.uniform(0.02,0.35),0.05+0.15*contact,actor_density]
            world_plus={'drivable_crop':[lane_width,clearance,lane_width/2+float(x_plus[1]),lane_width/2-float(x_plus[1]),float(abs(x_plus[1])/max(lane_width,1e-3)),friction,boundary_side,0.0],'future_occupancy':[actor_density,secondary_clear,max(0.0,5-secondary_clear),len(range(1,min(MAX_AGENTS,1+int(2+actor_density*8)))),base_speed,new_speed,route_dx,route_dy],'actor_flow':[route_dx,route_dy,base_speed,new_speed,yaw_reset,lat_shift,actor_density*base_speed,secondary_clear/max(base_speed,0.1)],'reachable_mask':[max(0.0,return_length),max(0.0,lane_width/2+x_plus[1]),max(0.0,lane_width/2-x_plus[1]),steering_scale,brake_scale,throttle_scale,friction,max(0.0,1-actor_density)],'goal_mask':[route_dx-float(x_plus[0]),route_dy-float(x_plus[1]),float(boundary_side),float(route_dx>float(x_plus[0])),lane_width,friction,1.0-damage_class/3,new_speed]}
            teacher_u=[]; teacher_traj=[]
            for k in range(RECOVERY_HORIZON):
                alpha=k/max(1,RECOVERY_HORIZON-1); u_delta=float((1-alpha)*(-0.25*np.sign(x_plus[1])*min(1,abs(x_plus[1])/max(lane_width,1e-3)))+0.10*alpha*(-steer)); u_brake=float(np.clip(0.15+0.35*(actor_density>0.55)+0.25*(r_sec<0),0,1)); u_throttle=float(np.clip(0.35*(r_return>0)+0.15*(friction>0.65)-0.20*u_brake,0,1)); teacher_u.append([u_delta,u_brake,u_throttle])
            for k in range(RECOVERY_HORIZON+1):
                alpha=k/max(1,RECOVERY_HORIZON); st=x_plus.copy(); st[0]+=alpha*max(1,0.25*return_length); st[1]*=(1-0.70*alpha); st[2]*=(1-0.50*alpha); st[3]=(1-alpha)*st[3]+alpha*min(13.9,max(2,new_speed+1)); st[4]*=(1-0.5*alpha); st[5]*=(1-0.6*alpha); st[6]*=(1-0.3*alpha); st[7]*=(1-0.3*alpha); st[8]*=(1-0.6*alpha); st[9:12]=teacher_u[min(k,RECOVERY_HORIZON-1)]; teacher_traj.append(st.tolist())
            audit={'has_contact':bool(contact),'event_type':event_type,'event_side':SIDE_NAMES[side_id],'normal_xy':normal.tolist(),'clearance':float(clearance),'impulse_proxy':float(rho_imp),'reset':z[11:18].tolist(),'affordance':z[18:24].tolist(),'degradation':d_deg.tolist(),'uncertainty':z[29:32].tolist()}
            row={'root_id':root_id,'split':split,'family':family,'action_id':int(a),'action_name':action_name,'action_vec':action_vec.tolist(),'o_hist':o_hist.tolist(),'h_ctx':h_ctx.tolist(),'rho_imp':rho_imp,'harm_bin':int(harm_bin),'x_t':x_minus.tolist(),'x_minus':x_minus.tolist(),'x_plus':x_plus.tolist(),'event_type':event_type,'event_time':float(duration),'deg':d_deg.tolist(),'world_plus':world_plus,'teacher_u':teacher_u,'teacher_traj':teacher_traj,'m_star':r_star.tolist(),'audit_mech':audit,'d_deg':d_deg.tolist(),'z_mech':z.tolist(),'r_star':r_star.tolist(),'b_star':int(np.argmin(r_star)),'s_star':float(np.min(r_star)),'calib_group':{'event_type':event_type,'contact_type':'contact' if contact else 'boundary','contact_side':SIDE_NAMES[side_id],'boundary_side':'right' if boundary_side>0 else 'left','friction_bin':'low' if friction<0.55 else ('mid' if friction<0.85 else 'high'),'damage_class':str(damage_class),'density_bin':'dense' if actor_density>0.65 else 'normal','town':town,'family':family}}
            root_rows.append(row)
        annotate_root_counterfactual_fields(root_rows, expected_actions=actions_per_root); rows.extend(root_rows)
    return rows

def write_jsonl(rows: Sequence[Dict], output: str|Path)->None:
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8') as f:
        
        def _default(o):
            import numpy as _np
            if isinstance(o, (_np.floating, _np.integer)):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
            raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')
        for row in rows: f.write(json.dumps(row,separators=(',',':'), default=_default)+'\n')

def main()->None:
    p=argparse.ArgumentParser(description='Generate clean synthetic MRVP dataset for smoke tests.'); p.add_argument('--output',required=True); p.add_argument('--n-roots',type=int,default=240); p.add_argument('--actions-per-root',type=int,default=NUM_ACTIONS); p.add_argument('--seed',type=int,default=7); args=p.parse_args(); rows=make_synthetic_rows(args.n_roots,args.actions_per_root,args.seed); write_jsonl(rows,args.output); print(json.dumps({'output':args.output,'rows':len(rows)}))
if __name__=='__main__': main()
