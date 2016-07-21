#!groovy

def katsdp = fileLoader.fromGit('scripts/katsdp.groovy', 'git@github.com:ska-sa/katsdpjenkins', 'master', 'katpull', '')
katsdp.standardBuild(maintainer: 'bmerry@ska.ac.za', cuda: true)
