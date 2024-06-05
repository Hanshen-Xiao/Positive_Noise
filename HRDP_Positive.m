clc;
clear;

T = 250000;
delta = 10^(-4);
Alpha = 25;
Scale = 1000;

%% Laplace 

% count = 0;
% start = 1;
% for R = 100:10:2000
%     R
%     mu = R/2;
% for sigma= start:0.5:start+0.5*Scale
%     prob_Lap = zeros(R+1);
%     for j = 0:R
%         prob_Lap(j+1) = exp(-abs(j-mu)/sigma);
%     end
%     prob_Lap = prob_Lap/(sum(prob_Lap));
%     if prob_Lap(1)>delta/(T) || prob_Lap(R+1)>delta/(T)
%         break;
%     end
%     count = count+1;
%     sm = 0;
%     for j=0:R
%         sm = sm+ j^2*prob_Lap(j+1);
%     end
%     SM_Lap(count) = sm;
%     
%     eps_0 = 1/sigma;
%     delta_0 = max(prob_Lap(1),prob_Lap(R+1));
%     Eps_adv(count) = sqrt(2*T*log(1/(delta-T*delta_0)))*eps_0+T*eps_0*(exp(eps_0)-1);
%     
%     Renyi = zeros(Alpha-1,1);
%     for alpha = 2:Alpha
%         Renyi_item_1 =0;
%         Renyi_item_2 =0;
%         for j =0:R-1
%             Renyi_item_1 = Renyi_item_1+ prob_Lap(j+2)^(alpha)/prob_Lap(j+1)^(alpha-1);
%             Renyi_item_2=  Renyi_item_2+ prob_Lap(j+1)^(alpha)/prob_Lap(j+2)^(alpha-1);
%         end
%         Renyi_item_1 = 1/(alpha-1)*(T*log(Renyi_item_1)+log(1/(delta-T*prob_Lap(1))));
%         Renyi_item_2 = 1/(alpha-1)*(T*log(Renyi_item_2)+log(1/(delta-T*prob_Lap(R+1))));
%         Renyi(alpha-1) = max(Renyi_item_1,Renyi_item_2);
%     end
%     Eps_Lap(count) = min(Renyi);
%     Parameter_Lap(count,:)= [sigma,R];
%     end
% end
% 
% C =1;
% for p = 2:0.25:10
%     p
%     A = find(Eps_Lap<p);
%     SM_fig(C) = min(SM_Lap(A));
%     R_fig(C)= min(Parameter_Lap(A,2));
%     B = find(Eps_adv<p);
%     SM_adv_fig(C) = min(SM_Lap(B));
%     R_adv_fig(C)= min(Parameter_Lap(B,2));
%     C = C+1; 
% end





%% Gaussian
% R = 500;
% 
% 
% count = 0;
% start = 1;
% scale = 2;
% Scale = 200;
% for R = 100:100:1000
%     R
% for sigma= start:scale:start+scale*Scale
%     for mu = R/2
%         prob_Gau = zeros(R+1);
%         for j = 0:R
%             prob_Gau(j+1) = exp(-(j-mu)^2/sigma^2);
%         end
%         prob_Gau = prob_Gau/(sum(prob_Gau));
%         if prob_Gau(1)>delta/T || prob_Gau(R+1)>delta/T
%             break;
%         end
%        
%         
%         Renyi = zeros(Alpha-1,1);
%         
%         for alpha = 2:Alpha
%             Renyi_item_1 =0;
%             Renyi_item_2 =0;
%             for j =0:R-1
%                 ratio_item = prob_Gau(j+2)/prob_Gau(j+1);
%     %             ratio_item = max(prob_Gau(j+2)/prob_Gau(j+1),10^10);
%     %             ratio_item = min(prob_Gau(j+2)/prob_Gau(j+1),10^(-10));
%                 Renyi_item_1 = Renyi_item_1+ prob_Gau(j+2)*ratio_item^(alpha-1);
%                 Renyi_item_2=  Renyi_item_2+ prob_Gau(j+1)*ratio_item^(-(alpha-1));
%             end
%             Renyi_item_1 = 1/(alpha-1)*(T*log(Renyi_item_1)+log(1/(delta-T*prob_Gau(1))));
%             Renyi_item_2 = 1/(alpha-1)*(T*log(Renyi_item_2)+log(1/(delta-T*prob_Gau(R+1))));
%             Renyi(alpha-1) = max(Renyi_item_1,Renyi_item_2);
%         end
%         if min(Renyi)<15
%             count= count+1;
%             Eps_Gau(count) = min(Renyi);
%             sm = 0;
%             for j=0:R
%                 sm = sm+ j^2*prob_Gau(j+1);
%             end
%             SM_Gau(count) = sm;
%             Parameter(count,:) = [mu,sigma,R];
%         end
%     end
% end
% end
% 
% for i = 2:0.5:10
%     i
%     A = find(Eps_Gau<i);
%     min(SM_Gau(A))
% end


%% Trapezoid 
% RU = 500;
% 
% 
count = 0;
start = 50;
scale = 20;
Scale = 50;
scale_0 = 250
Scale_0 = 20;
for R = 25000
    R
for sigma= start:scale:start+scale*Scale
    sigma 
    for sigma_0 =  start:scale_0:start+scale_0*Scale_0
        sigma_0
        for a = 500:1000:10000
            for b = R-10000:1000:R-500
                if b<a 
                    break;
                end
            
                prob_T = zeros(R+1);
                for j = 0:R
                    if j<a
                        prob_T(j+1) = exp(-(j-a)^2/sigma^2);
                    elseif j>b
                        prob_T(j+1) = exp(-(j-b)^2/(sigma)^2);
                    else
                        prob_T(j+1) = exp(-(j-(a+b)/2)^2/(sigma_0)^2);
                    end
                end
                prob_T = prob_T/(sum(prob_T));
                if prob_T(1)>delta/T || prob_T(R+1)>delta/T
                    break;
                end


                Renyi = zeros(Alpha-1,1);

                for alpha = 2:Alpha
                    Renyi_item_1 =0;
                    Renyi_item_2 =0;
                    for j =0: R-1
                        ratio_item = prob_T(j+2)/prob_T(j+1);
            %             ratio_item = max(prob_Gau(j+2)/prob_Gau(j+1),10^10);
            %             ratio_item = min(prob_Gau(j+2)/prob_Gau(j+1),10^(-10));
                        Renyi_item_1 = Renyi_item_1+ prob_T(j+2)*ratio_item^(alpha-1);
                        Renyi_item_2=  Renyi_item_2+ prob_T(j+1)*ratio_item^(-(alpha-1));
                    end
                    Renyi_item_1 = 1/(alpha-1)*(T*log(Renyi_item_1)+log(1/(delta-T*prob_T(1))));
                    Renyi_item_2 = 1/(alpha-1)*(T*log(Renyi_item_2)+log(1/(delta-T*prob_T(R+1))));
                    Renyi(alpha-1) = max(Renyi_item_1,Renyi_item_2);
                end

                if min(Renyi)< 10
                    count= count+1;
                    Eps_T(count) = min(Renyi);
                    sm = 0;
                    for j=0:R
                        sm = sm+ j^2*prob_T(j+1);
                    end
                    SM_T(count) = sm;
                    Parameter_T(count,:) = [a,b,sigma,sigma_0,R];
                end
            end
            end
        end
end
end

C =1;
for p = 4 :0.25:10
    p
    A = find(Eps_T<p);
    SM_fig(C) = min(SM_T(A));
    R_fig(C)= min(Parameter_T(A,5));
    C = C+1; 
end


