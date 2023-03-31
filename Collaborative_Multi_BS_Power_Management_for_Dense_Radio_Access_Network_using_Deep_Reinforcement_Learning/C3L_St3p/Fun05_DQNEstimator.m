classdef Fun05_DQNEstimator < handle
	properties (SetAccess = private)
        SAEInfo;
        alpha;        
        weights;        
        hidden_layer;
    end
	
	methods
		function obj = Fun05_DQNEstimator(SAEInfo,alpha,hidden_layer)
            obj.SAEInfo = SAEInfo;
            obj.alpha = alpha;            
            obj.hidden_layer = hidden_layer;            
            obj.weights.input = normrnd(0,1,[SAEInfo.n_complexFeatures+1, hidden_layer(1)])/sqrt(obj.SAEInfo.n_complexFeatures);
            obj.weights.hidden = normrnd(0,1,[hidden_layer(1)+1, hidden_layer(2)])/sqrt(hidden_layer(1));
			obj.weights.out = normrnd(0,1,[hidden_layer(2)+1, length(obj.SAEInfo.RANInfo.v_actSpace)])/sqrt(hidden_layer(2));
        end
        
        function setWeights(obj,weights)
            obj.weights = weights;
        end
        
        function [m_Feartures m_hidden_out_value m_hidden_out_value2 m_outValue] = predict(obj,state)
            m_Feartures = obj.SAEInfo.getComplexStateFeatures(state);%features are already scaled.
            v_RCFeartures=size(m_Feartures);
            
            m_outValue=zeros(v_RCFeartures(2), length(obj.SAEInfo.RANInfo.v_actSpace));
            m_hidden_out_value=zeros(v_RCFeartures(2), obj.hidden_layer(1));
            m_hidden_out_value2=zeros(v_RCFeartures(2), obj.hidden_layer(2));
            for iR=1:v_RCFeartures(2)
                v_Feartures=[1 obj.SAEInfo.RANInfo.v_RC_stateSpace(1)*(m_Feartures(2,iR)-1)+m_Feartures(1,iR)];
                value.hidden_in_value = sum(obj.weights.input(v_Feartures,:));   %1 40t1=toc;
                value.hidden_out_value = 1./(1+exp(-value.hidden_in_value));     %1 40
                m_hidden_out_value(iR,:)=value.hidden_out_value;
                
                value.hidden_in_value2 = [1 value.hidden_out_value] * obj.weights.hidden;   %1 40
                value.hidden_out_value2 = 1./(1+exp(-value.hidden_in_value2));     %1 40
                m_hidden_out_value2(iR,:)=value.hidden_out_value2;
                            
                value.out_value = [1 value.hidden_out_value2] * obj.weights.out;    % 1 4
                m_outValue(iR,:)=value.out_value;
            end
        end
        
        function [losVal]=update(obj,state,action,target)
            n_v_RC_stateSpace=obj.SAEInfo.RANInfo.v_RC_stateSpace(1)*obj.SAEInfo.RANInfo.v_RC_stateSpace(2);   
            [m_feartures hidden_out_value hidden_out_value2 value] = obj.predict(state);
            action=round((action-obj.SAEInfo.RANInfo.v_actSpace(1))/(obj.SAEInfo.RANInfo.v_Parameter(3)))+1;
            out_value = value(action);
            vt_input=zeros(1,obj.hidden_layer(1));
            for i=1:obj.hidden_layer(1)                
                vt_input(i)=(out_value - target) * ...%1 1
                                     sum(obj.weights.out(2:end,action)' .* ...%1 40
                                     (hidden_out_value2.*(1-hidden_out_value2)) .* ...%[1 40]*[1 40] ->1 40
                                     obj.weights.hidden(i+1,:)) * ...%[1 40]*[40 1] -> 1 1
                                     hidden_out_value(i) * (1-hidden_out_value(i));                  
            end
            derivative_in(n_v_RC_stateSpace+1, obj.hidden_layer(1))=0;
            derivative_in(1,:)=vt_input;
            derivative_in(1+obj.SAEInfo.RANInfo.v_RC_stateSpace(1)*(m_feartures(2)-1)+m_feartures(1),:)=vt_input;
            obj.weights.input = obj.weights.input - obj.alpha * derivative_in;
            derivative_hidden(obj.hidden_layer(2)+1, obj.hidden_layer(2)) = 0;
            for i=1:obj.hidden_layer(2)
                derivative_hidden(:,i) = (out_value - target) * obj.weights.out(i+1) * ...
                                          hidden_out_value2(i) * (1-hidden_out_value2(i)) * [1 hidden_out_value];
                obj.weights.hidden(:,i) = obj.weights.hidden(:,i) - obj.alpha * derivative_hidden(:,i);                    
            end
            derivative_out(:,1) = (out_value- target) * [1 hidden_out_value2];
            obj.weights.out(:,action) = obj.weights.out(:,action) - obj.alpha * derivative_out;
            losVal=out_value- target;
        end

    end
end
