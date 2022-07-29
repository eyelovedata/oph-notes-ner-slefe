--try to generate messily "labeled" notes - labeled by their fields 
--sle and fundus grouped and combined (N=56029)
select examparsed.smartformid, max(extod) as extod, max(extos) as extos, 
max(sleodll) as sleodll, max(sleosll) as sleosll, max(sleodcs) as sleodcs, max(sleoscs) as sleoscs, 
max(sleodk) as sleodk, max(sleosk) as sleosk, max(sleodac) as sleodac, max(sleosac) as sleosac, max(sleodiris) as sleodiris, max(sleosiris) as sleosiris, 
max(sleodlens) as sleodlens, max(sleoslens) as sleoslens, max(sleodvit) as sleodvit, max(sleosvit) as sleosvit, 
max(feoddisc) as feoddisc, max(feosdisc) as feosdisc, max(feodcdr) as feodcdr, max(feoscdr) as feoscdr, max(feodmac) as feodmac, max(feosmac) as feosmac, 
max(feodvess) as feodvess, max(feosvess) as feosvess, max(feodperiph) as feodperiph, max(feosperiph) as feosperiph, 
examfield.DATE_OF_SERVICE, examfield.PROVIDER_DEID as provider_deid, examfield.pat_deid, notes.NOTE
from examparsed, examfield, notes
where examparsed.smartformid=examfield.smartformid
and examfield.pat_deid = notes.pat_deid 
and examfield.date_of_service = substr(notes.ENCOUNTER_DATE, 0, 10)
and examparsed.sleodll not null
and notes.NOTE_DESC='Progress Notes'
and length(note)>2000
group by note_deid 

