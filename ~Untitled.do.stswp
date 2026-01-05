clear all
use "C:\Users\32489\Dropbox\Family Insurance\delete.dta" 

gen event_time_PER_treat100=event_time_PER_treat+100
gen event_time100=event_time+100

gen treat=0
replace treat=1 if treat_group==1 & event_time>=0

gen ls=log( wife_share)


*areg wife_share b99.event_time100 A,absorb(age agei event_time iz )

*areg wife_share treat,absorb(age event_time100  idd )

*areg ls b100.event_time_PER_treat100 ,absorb(age event_time100  idd )


reg wife_share b99.event_time_PER_treat100 b99.event_time100 treat_group
areg wife_share b99.event_time_PER_treat100, absorb(event_time100 treat_group age)




reg wife_share treat_group post inter