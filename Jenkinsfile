#!groovy
@Library('katsdpjenkins') _
katsdp.killOldJobs()
katsdp.setDependencies([
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katpoint/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master',
    'ska-sa/katsdpdockerbase/master'])
katsdp.standardBuild(cuda: true)
katsdp.mail('sdpdev+katcbfsim@ska.ac.za')
