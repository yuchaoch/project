classdef Fun05_SAEnvironment < handle

    properties (SetAccess=private)
        RANInfo;
        
        curState;
        n_complexFeatures;
        m_EgySvTGOLosReward;
    end
    
    methods
        function obj=Fun05_SAEnvironment(RANInfo)
            obj.RANInfo=RANInfo;
            vt_atvCell_RSRP=round((obj.RANInfo.v_atvCell_RSRP)/obj.RANInfo.v_Parameter(3))*obj.RANInfo.v_Parameter(3);
            obj.curState = [obj.RANInfo.m_traficPoson(1,RANInfo.v_atvCell); vt_atvCell_RSRP];
            obj.n_complexFeatures=obj.RANInfo.v_RC_stateSpace(1)*obj.RANInfo.v_RC_stateSpace(2);
        end
        
        function curState=getCurState(obj)
            obj.RANInfo.setPreTrafic([],[],[]);
            obj.RANInfo.decTTIofTraficPoson();
            obj.RANInfo.updateAtvCellUser();
            obj.RANInfo.calPerformanceTrafic([]);
            vt_atvCell_RSRP=round((obj.RANInfo.v_atvCell_RSRP)/obj.RANInfo.v_Parameter(3))*obj.RANInfo.v_Parameter(3);
            curState = [obj.RANInfo.m_traficPoson(1,obj.RANInfo.v_atvCell); vt_atvCell_RSRP];
            obj.curState=curState;
        end
        
        function m_Feartures = getComplexStateFeatures(obj,state)            
            m_Feartures=state;
            m_Feartures(1,:)=round((state(1,:)-obj.RANInfo.v_TrafficSpace(1))/obj.RANInfo.v_Parameter(3))+1;
            m_Feartures(2,:)=round((state(2,:)-obj.RANInfo.v_RSRPSpace(1))/obj.RANInfo.v_Parameter(3))+1;
        end
        function [m_nextState m_EgySvTGOLosReward v_Done] = step(obj, v_Act)
            n_Act=length(v_Act);
            
            obj.RANInfo.calPerformanceTrafic([]);
            vt_atvCell_TGO=obj.RANInfo.v_atvCell_TGO;
            v_atvCell_ITF=obj.RANInfo.v_atvCell_ITF;
            v_atvCell_RSRP=obj.RANInfo.v_atvCell_RSRP;
            obj.RANInfo.calPerformanceTrafic(v_Act);
            v_atvCell_LPTGO=obj.RANInfo.v_atvCell_TGO;
            v_atvCell_LPITF=obj.RANInfo.v_atvCell_ITF;
            v_atvCell_LPRSRP=obj.RANInfo.v_atvCell_RSRP-v_Act;
            v_EgySv=1-1./(10.^(v_Act/10));
            v_TGOLos=vt_atvCell_TGO./(v_atvCell_LPTGO+0.00001)-1;
            vt_idxTGOLos=find(0==v_TGOLos);            v_TGOLos(vt_idxTGOLos)=0.0001;
            v_Reward=v_TGOLos./(15.2-v_Act);
            v_Done=double(v_TGOLos<=0);
            m_EgySvTGOLosReward=[v_TGOLos; v_EgySv; v_Reward; vt_atvCell_TGO; vt_atvCell_TGO-v_atvCell_LPTGO];
            m_EgySvTGOLosReward=[m_EgySvTGOLosReward; v_atvCell_RSRP; v_atvCell_LPRSRP; v_atvCell_ITF; v_atvCell_LPITF];
            obj.m_EgySvTGOLosReward=m_EgySvTGOLosReward;            
            obj.RANInfo.decTTIofTraficPoson();
            vt_atvCell_RSRP=round((obj.RANInfo.v_gridRSRP(obj.RANInfo.v_atvGrid)')/obj.RANInfo.v_Parameter(3))*obj.RANInfo.v_Parameter(3);
            m_nextState= [obj.RANInfo.m_traficPoson(1,obj.RANInfo.v_atvCell); vt_atvCell_RSRP];
        end
        
        function m_policyProb=EpsilonGreedyPolicy(obj, epsilon, m_outValue)            
            v_RCOutValue=size(m_outValue);
            m_policyProb=zeros(v_RCOutValue);
            for iR=1:v_RCOutValue(1)
                v_policyProb(1:v_RCOutValue(2))=epsilon*(1/length(obj.RANInfo.v_actSpace));
                rndEpsilon=rand;
                if rndEpsilon>epsilon
                    v_outValue=m_outValue(iR,:);
                    [val idx]=max(v_outValue);
                    v_idx=find(val==v_outValue);                    
                    if isempty(v_idx)
                        v_policyProb(idx)=1-epsilon;
                    else
                        idx=v_idx(1);
                        v_policyProb(idx)=(1-epsilon)*obj.RANInfo.v_actSpaceProb(idx);
                    end
                end
                m_policyProb(iR,:)=v_policyProb;
            end
        end
        
        function memory_buffer = updateReplayMemory(obj, memory_buffer, n_memory, replayMemory, n_atvCell)            
            for iR=1:n_atvCell
                memory_buffer(2:n_memory) = memory_buffer(1:n_memory-1);                
                memory_buffer(1).state = replayMemory.state(:,iR);
                memory_buffer(1).action = replayMemory.action(iR);
                memory_buffer(1).next_state = replayMemory.next_state(:,iR);
                memory_buffer(1).reward = replayMemory.reward(iR);
                memory_buffer(1).done = replayMemory.done(iR);
            end
            obj.RANInfo.decTraficPoson();
            obj.RANInfo.setPreTrafic(obj.RANInfo.m_traficPoson, obj.RANInfo.v_atvCell, obj.RANInfo.v_atvGrid)
            obj.RANInfo.updateAtvCellUser();
            obj.RANInfo.calPerformanceTrafic([]);
            vt_atvCell_RSRP=round((obj.RANInfo.v_atvCell_RSRP)/obj.RANInfo.v_Parameter(3))*obj.RANInfo.v_Parameter(3);
            obj.curState=[obj.RANInfo.m_traficPoson(1,obj.RANInfo.v_atvCell); vt_atvCell_RSRP];            
        end
    end
    
end