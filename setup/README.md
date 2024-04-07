# Setup

## Log onto Compute Node

If you haven't already, please make sure to download the [GT VPN](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042139) and to verify that you have an ``ssh`` installation. After you've done this, please enter your terminal environment and run the following code, where ``<username>`` is your GT username.
```bash
ssh <username>@login-ice.pace.gatech.edu
```
If this runs successfully, you will be able to enter your password. Note that your password will not be visible from your end. If you've successfully entered the repository, you will see a prompt that looks like the following. 

```
This computer system is the property of Georgia Institute of
Technology.  Any user of this system must comply with all Institute
and Board of Regents policies, including the Acceptable Use
Policy (AUP), Data Privacy Policy (DPP) and Cyber Security
Policy (CSP), see http://b.gatech.edu/it-policies.  Users should
have no expectation of privacy, as any and all files on this system
may be intercepted, monitored, copied, inspected, and/or disclosed to
authorized personnel in order to meet institute obligations.

By using this system, I acknowledge and consent to these terms.

#######################################
# Welcome to GT Instructional Cluster #
#######################################

If you require assistance with this system, please contact your
course instructor or teaching assistant (TA).

[<username>@login-ice-1 ~]$ 
```

If you see this prompt, you've successfully logged on to the cluster! Logging onto other clusters (e.g. Hive, Phoenix) is simply a matter of changing the login node, which in this case is associated with PACE's ICE (Instructional Computing Environment). 

After you've successfully logged on, please run the following code to fork the repository that you'll be using today. 
```
cd
git clone https://github.com/suco-gt/HPC-workshop-2/    
ls | grep workshop 
```
If the repository is present, this should print the repository name. If not, please let us know. 

At this point, you've successfully logged into the cluster. 

## Open a GPU Node

Once you've logged onto the cluster, run the following command to verify that you're able to open a GPU node. Note that finding an available node may not be possible, so the process may take a while. Fill in the ``[GPU_name]`` section with ```H100, A100, V100, RTX6000, A40```.

```
salloc --nodes=1 --ntasks-per-node=1 --gres=gpu:[GPU_name]:1 --mem=0 --time=0:15:00
```
