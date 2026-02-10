*------------------------------------------------------------------------------*
*							Globals 
*------------------------------------------------------------------------------*

	if  "`c(username)'" == "sara" {
		global JPSC 		"~/Dropbox/Family Insurance"
	}
	// modify yours appropriately
	else if "`c(username)'" == "32489" {
		global JPSC 		"~/Dropbox/Family Insurance"
	}

		
		global cleandata	"${JPSC}/Empirical analysis//Data/JPSC2022_v2/cleandata"
		global tables 		"${JPSC}/Tables"
		global figures 		"${JPSC}/Figures"

*------------------------------------------------------------------------------*

*------------------------------------------------------------------------------*
*	    Import data, sample selection and variables creation 
*------------------------------------------------------------------------------*

*import data
clear all 
use "${JPSC}/simulated_did.dta" , clear


*Sample selection
keep if power>=0 & age>=agei  & event_time>=-5 & event_time<=10 & agei-20<=age_policy-1

*Treatment group
gen treat_group=0
replace treat_group=1 if age_policy+20>=30

*Treatment per event
gen event_time_PER_treat=event_time*treat_group 

*Modify variables to avoid fixed effects with negative value
gen event_time_PER_treat100=event_time_PER_treat+100
gen event_time100=event_time+100

*Dependent variable
gen log_wife_share=log( wife_share)

*------------------------------------------------------------------------------*
*							Empirical Analysis
*------------------------------------------------------------------------------*

*Did regression
areg log_wife_share b99.event_time_PER_treat100 if event_time100>95, absorb(event_time100 age idd)
