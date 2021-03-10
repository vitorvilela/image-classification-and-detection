# Windows Management Instrumentation - Running from Windows
#import wmi 
import wmi_client_wrapper as wmi
import time
import threading
import os



def service_maintenance(currentProcessId, c):
  
  # The maintenance file must have just one command word
  # stop - to stop service program
  # start - to start service program
  # To start a process, make sure you finish it before
  
  filesPath = os.getcwd()
  commandFileName = 'maintenance.mc'  
  fullSentinelPath = os.path.join(filesPath, commandFileName)
  serviceFileName = 'service.py'
  serviceCommandLine = 'python ' + os.path.join(filesPath, serviceFileName)

  while True:  
    
    # Wait 10 minutes to proceed with maintenance
    time.sleep(600)
    maintenanceFile = open(fullSentinelPath, 'r') 
    maintenanceCommand = maintenanceFile.read()
    maintenanceFile.close()
    
    if maintenanceCommand == 'start':
      launched_id, _ = c.Win32_Process.Create(CommandLine=serviceCommandLine)
      return launched_id
    
    elif maintenanceCommand == 'stop':
      for process in c.Win32_Process(name='python3'):
        if process.ProcessId == currentProcessId:
          _ = process.Terminate()
      
      
      
def keep_service_operation(currentProcessId, c): 
  
  filesPath = os.getcwd()
  serviceFileName = 'service.py'
  serviceCommandLine = 'python ' + os.path.join(filesPath, serviceFileName)
  
  while True:    
  
    # Check if service is properly running on every 60 seconds
    time.sleep(60)

    for process in c.Win32_Process(name='python3'):
      if process.State == 'Stopped' and process.ProcessId == currentProcessId:

        statsFile = open('breakingStats.mb', 'a')   
        phrase = str(currentProcessId) + '\n'
        statsFile.write(phrase)
        statsFile.close()
                
        launched_id, _ = c.Win32_Process.Create(CommandLine=serviceCommandLine)
               
        return launched_id


def main():

  c = wmi.WMI()

  # Abrir na inicialização do Windows
  launched_id, _ = c.Win32_Process.Create(CommandLine='python servicen.py')

  maintenanceThread = threading.Thread(target=service_maintenance, args=(launched_id, c,))
  operationThread = threading.Thread(target=keep_service_operation, args=(launched_id, c,))

  maintenanceThread.start()
  operationThread.start()

  # Never join
  #maintenanceThread.join()
  #operationThread.join()


if __name__ == '__main__':     
  main()
  