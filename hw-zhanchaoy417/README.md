<!-- ![documentation](https://github.com/coloradointroai/hw_spring_2022/workflows/documentation/badge.svg) -->
<p align="center">
<a href=""><img alt="https://github.com/coloradointroai/hw_spring_2022" src="https://avatars.githubusercontent.com/u/98675389?s=200&v=4" width="200" height="200" /></a>
<br><br>
<a href="https://coloradointroai.github.io/colorado_intro_ai.html"><img height="20" alt="documentation" src="https://github.com/coloradointroai/hw_spring_2022/workflows/documentation/badge.svg"></a>
</p>

# Colorado Intro to AI

## **Recommended Setup**

1. Clone this repository onto your local machine: [how clone works](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
   
   * On Ubuntu 20.04 the command `git clone https://github.com/coloradointroai/hw-<your user name and remove the <> on both sides>.git` will make a copy of all files in this repo into where you call the `git clone` command  
2. Install all required python package dependancies for `colorado_intro_ai` with the command `pip install -r requirements.txt` **We recommend that you call this function within a python [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)**

   * For installation see [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)
   * For how to start a virtualenv and install requirements.txt see this [post on stackoverflow](https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from)
3. DO THIS BEFORE YOU START TO MOVE YOUR CODE TO ANY CLASS FILES: In order to automatically load in changes that you made to any colorado_intro_ai class you need to edit your ipython configuration.
   * See this post for how to [reaload ipython submodules](https://stackoverflow.com/questions/5364050/reloading-submodules-in-ipython)




### Documentation Preview

* Documentation is generated using [pdoc](https://github.com/mitmproxy/pdoc). Anytime a push is made to the master branch of this repository documentation is automatically generated using a [custom github workflow](https://github.com/coloradointroai/hw_spring_2022/blob/master/.github/workflows/documentation.yml). When the workflow finishes documentation is available at [coloradointroai.github.io](https://coloradointroai.github.io). 
* If you want to preview documentation changes on your own machine run `pdoc ./colorado_intro_ai` and a window should open that automatically updates when you save changes to any file in the `colorado_intro_ai` directory. 
