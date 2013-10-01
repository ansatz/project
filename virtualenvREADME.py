####################################################3
#***overkill***
#http://stackoverflow.com/questions/4324558/whats-the-proper-way-to-install-pip-virtualenv-and-distribute-for-python
curl -Lo virtualenv-tmp.tar.gz 'https://github.com/pypa/virtualenv/tarball/master'
tar xzf  >>`dir/` 
python virtualenv.py py-env0
rm -rf `dir`
py-env0/bin/pip install virtualenv.tar.gz
#bootstrap new env
py-env0/bin/virtualenv py-env1
py-env0/bin/virtualenv py-env2
######################################################


#http://docs.python-guide.org/en/latest/dev/virtualenvs.html

#will set up newsfeed python rss reader
#http://home.arcor.de/mdoege/newsfeed/

#=== virtualenv ===
pip install virtualenv

#**usage**
#1. create virtual environment	<br>
#creates copy of Python in venv	<br>
cd `working-dir`
virtualenv venv
#2. to begin using virtual environment ACTIVATE
source venv/bin/activate

#...now install any new modules without affecting system python or other virtual environments

# if done with environment DEACTIVATE
deactivate

# to delete a virtual environment
rm `working-dir`

#=== virtualenvwrapper ===
#places all virtual environements one place
#provides commands
mkvirtualenv `an-environment`
lssitepackages
ls WORKON_HOME
workon `an-environemtn` 
postmkvirtualenv

#**install**
#1.make sure virtualenv is first installed
#http://virtualenvwrapper.readthedocs.org/en/latest/#introduction
pip install virtualenvwrapper
export WORKON_HOME=~/Envs
source /usr/local/bin/virtualenvwrapper.sh

#**usage**
#create env -- creates venv/ in ~/Envs
mkvirtualenv `venv`
#work on a virtual env
#provides tab-complete, and deactivates other env, so quick to switch
workon venv
#deactivate
deactivate
#to remove
rmvirtualenv venv


