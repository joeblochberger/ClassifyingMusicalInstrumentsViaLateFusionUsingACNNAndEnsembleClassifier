%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%HelperSegmentedMelSpectrograms
function X = HelperSegmentedMelSpectrograms_JABmono(x,fs,varargin)
% For mono wav files

p = inputParser;
addParameter(p,'WindowLength',2048*2);
addParameter(p,'HopLength',1024*2);
addParameter(p,'NumBands',128*2);
% addParameter(p,'SegmentLength',1);
% addParameter(p,'SegmentOverlap',0);
addParameter(p,'FFTLength',2*2048*2);
parse(p,varargin{:})
params = p.Results;

x = mean(x,2);
x = x./max(max(x));

spec = melSpectrogram(x,fs, ...
    'Window',hann(params.WindowLength,'periodic'), ...
    'OverlapLength',round(0.5*params.WindowLength), ...
    'FFTLength',params.FFTLength, ...
    'NumBands',params.NumBands, ...
    'FrequencyRange',[0,floor(fs/2)]);
spec = log10(spec+eps);

X = reshape(spec,size(spec,1),size(spec,2),size(x,2),[]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%