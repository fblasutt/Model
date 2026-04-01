clear all
use "C:\Users\32489\Dropbox\Family Insurance\delete.dta" 



gen event100=event_time+100
gen event100m=event100*treat_group

reghdfe wife_share b99.event100 b99.event100m, absorb( age idd)

***************************
*Or do it Sara style below
***************************


********** Restrictions

global start_event_window  		-4
global end_event_window			4

* keep x years before and x years after the reform
gen withinWindow =  (event_time >= $start_event_window & event_time <= $end_event_window)

* flag those who were always couples for x years before and x years after the reform
*bys idd: egen help = sum(missing(wife_share))
bys idd: egen help = sum(power<0)  if withinWindow & event_time<0
bys idd: egen help2 = mean(help)
gen always_married = (help2==0)
drop help*



*keep if always_married

********** Generate event dummies

tab event_time if withinWindow, gen(t)
forvalues i=1/ 9  {
	replace t`i' = 0 if event_time < $start_event_window | event_time >  $end_event_window
}
* outside window indicators
gen far_pre  = (event_time < $start_event_window )
gen far_post = (event_time >  $end_event_window )


 
tab treat_group, gen(leng)

**** Multiply with time to event dummies
forvalues i=1/ 9  {
		gen t`i'_leng2 = t`i'*leng2
}

gen far_pre_leng2 = far_pre*leng2
gen far_post_leng2 = far_post*leng2





gen ls=log( wife_share)



*reghdfe wife_share far_pre t1-t3 t5-t9 far_post far_pre_leng2 t1_leng2-t3_leng2 t5_leng2-t9_leng2 far_post_leng2, absorb( age idd)


