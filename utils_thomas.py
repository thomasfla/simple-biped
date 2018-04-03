def restert_viewer_server(procName='gepetto-gui', delay = 0.2):
    ''' This function kill and restart the viewer server'''
    import os, signal, subprocess
    from time import sleep
    def check_kill_process(pstring):
        for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            os.kill(int(pid), signal.SIGKILL)
        check_kill_process(procName)
    sleep(delay)
    proc = subprocess.Popen([procName], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    sleep(delay)
