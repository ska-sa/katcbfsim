#!groovy

def katsdp = fileLoader.fromGit('scripts/katsdp.groovy', 'git@github.com:ska-sa/katsdpjenkins', 'master', 'katpull', '')
katsdp.setDependencies([
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katpoint/master',
    'ska-sa/katsdpdockerbase/master'])
katsdp.standardBuild(maintainer: 'bmerry@ska.ac.za', cuda: true)
