#!groovy
@Library('katsdpjenkins') _
katsdp.killOldJobs()
katsdp.setDependencies([
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katpoint/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master',
    'ska-sa/katsdpdockerbase/new-rdma-core'])
katsdp.standardBuild(cuda: true, python3: true, python2: false,
                     katsdpdockerbase_ref: 'new-rdma-core')
katsdp.mail('sdpdev+katcbfsim@ska.ac.za')
