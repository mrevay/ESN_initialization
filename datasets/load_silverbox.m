function [u_train, y_train, u_test, y_test] = load_silverbox()
%LOAD_SILVERBOX Summary of this function goes here
%   Detailed explanation goes here
data = load('./datasets/SilverboxFiles/SilverboxFiles/SNLS80mV.mat');
u_train = data.V1(1, 40500:end);
y_train = data.V2(1, 40500:end);

u_test = data.V1(1, 1:40500);
y_test = data.V2(1, 1:40500);

end

