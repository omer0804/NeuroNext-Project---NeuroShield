// NeuroShield - iOS App
// ViewController.swift

import UIKit
import WatchConnectivity
import Charts

class ViewController: UIViewController {
    
    // UI Elements
    private let statusLabel = UILabel()
    private let heartRateLabel = UILabel()
    private let watchConnectionLabel = UILabel()
    private let nightmareRiskLabel = UILabel()
    private let sleepStateLabel = UILabel()
    private let heartRateChart = LineChartView()
    private let startMonitoringButton = UIButton(type: .system)
    private let stopMonitoringButton = UIButton(type: .system)
    private let remHistoryButton = UIButton(type: .system)
    
    // Data visualization
    private var heartRates: [Double] = []
    private var timestamps: [Double] = []
    private var isMonitoring = false
    
    // REM sleep windows history
    private var remWindows: [NightmareDetectionManager.REMSleepWindow] = []
    
    // User ID
    private let userId = "user_\(Int(Date().timeIntervalSince1970))"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupNotifications()
        checkWatchConnection()
        
        // Set the user ID in the NightmareDetectionManager
        NightmareDetectionManager.shared.setUserId(userId)
    }
    
    private func setupUI() {
        view.backgroundColor = .systemBackground
        title = "NeuroShield"
        
        // Status Label
        statusLabel.text = "Status: Not Monitoring"
        statusLabel.font = .boldSystemFont(ofSize: 18)
        statusLabel.textAlignment = .center
        
        // Heart Rate Label
        heartRateLabel.text = "Heart Rate: --"
        heartRateLabel.font = .systemFont(ofSize: 16)
        heartRateLabel.textAlignment = .center
        
        // Watch Connection Label
        watchConnectionLabel.text = "Watch: Disconnected"
        watchConnectionLabel.font = .systemFont(ofSize: 16)
        watchConnectionLabel.textAlignment = .center
        
        // Sleep State Label
        sleepStateLabel.text = "Sleep State: Awake"
        sleepStateLabel.font = .systemFont(ofSize: 16)
        sleepStateLabel.textAlignment = .center
        
        // Nightmare Risk Label
        nightmareRiskLabel.text = "Nightmare Risk: Low"
        nightmareRiskLabel.font = .boldSystemFont(ofSize: 18)
        nightmareRiskLabel.textAlignment = .center
        nightmareRiskLabel.textColor = .systemGreen
        
        // Chart
        heartRateChart.noDataText = "No heart rate data available"
        heartRateChart.chartDescription.enabled = false
        heartRateChart.rightAxis.enabled = false
        heartRateChart.xAxis.labelPosition = .bottom
        heartRateChart.legend.enabled = false
        
        // Buttons
        startMonitoringButton.setTitle("Start Monitoring", for: .normal)
        startMonitoringButton.backgroundColor = .systemBlue
        startMonitoringButton.setTitleColor(.white, for: .normal)
        startMonitoringButton.layer.cornerRadius = 10
        startMonitoringButton.addTarget(self, action: #selector(startMonitoring), for: .touchUpInside)
        
        stopMonitoringButton.setTitle("Stop Monitoring", for: .normal)
        stopMonitoringButton.backgroundColor = .systemRed
        stopMonitoringButton.setTitleColor(.white, for: .normal)
        stopMonitoringButton.layer.cornerRadius = 10
        stopMonitoringButton.isEnabled = false
        stopMonitoringButton.alpha = 0.5
        stopMonitoringButton.addTarget(self, action: #selector(stopMonitoring), for: .touchUpInside)
        
        remHistoryButton.setTitle("View REM History", for: .normal)
        remHistoryButton.backgroundColor = .systemPurple
        remHistoryButton.setTitleColor(.white, for: .normal)
        remHistoryButton.layer.cornerRadius = 10
        remHistoryButton.addTarget(self, action: #selector(showREMHistory), for: .touchUpInside)
        
        // Add to view and setup constraints
        let stackView = UIStackView(arrangedSubviews: [
            statusLabel, 
            heartRateLabel, 
            watchConnectionLabel,
            sleepStateLabel,
            nightmareRiskLabel,
            heartRateChart,
            startMonitoringButton,
            stopMonitoringButton,
            remHistoryButton
        ])
        
        stackView.axis = .vertical
        stackView.spacing = 15
        stackView.distribution = .fill
        stackView.alignment = .fill
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(stackView)
        
        // Set constraints
        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            stackView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            stackView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            stackView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            
            heartRateChart.heightAnchor.constraint(equalToConstant: 200),
            startMonitoringButton.heightAnchor.constraint(equalToConstant: 50),
            stopMonitoringButton.heightAnchor.constraint(equalToConstant: 50),
            remHistoryButton.heightAnchor.constraint(equalToConstant: 50)
        ])
    }
    
    private func setupNotifications() {
        // Register for nightmare detection notifications
        NotificationCenter.default.addObserver(self, 
                                              selector: #selector(handleNightmareDetection), 
                                              name: .nightmareDetected, 
                                              object: nil)
        
        // Register for watch connectivity changes
        NotificationCenter.default.addObserver(self, 
                                              selector: #selector(checkWatchConnection), 
                                              name: .watchConnectivityChanged, 
                                              object: nil)
        
        // Register for heart rate updates
        NotificationCenter.default.addObserver(self, 
                                              selector: #selector(updateHeartRate(_:)), 
                                              name: .heartRateUpdated, 
                                              object: nil)
        
        // Register for REM sleep state changes
        NotificationCenter.default.addObserver(self,
                                              selector: #selector(updateSleepState(_:)),
                                              name: .remSleepStateChanged,
                                              object: nil)
        
        // Register for new REM windows
        NotificationCenter.default.addObserver(self,
                                              selector: #selector(newREMWindowDetected(_:)),
                                              name: .newREMWindowDetected,
                                              object: nil)
    }
    
    @objc private func checkWatchConnection() {
        if WCSession.isSupported() {
            let session = WCSession.default
            
            if session.activationState == .activated && session.isPaired && session.isReachable {
                watchConnectionLabel.text = "Watch: Connected"
                watchConnectionLabel.textColor = .systemGreen
                startMonitoringButton.isEnabled = true
                startMonitoringButton.alpha = 1.0
            } else {
                watchConnectionLabel.text = "Watch: Disconnected"
                watchConnectionLabel.textColor = .systemRed
                startMonitoringButton.isEnabled = false
                startMonitoringButton.alpha = 0.5
            }
        } else {
            watchConnectionLabel.text = "Watch: Not Supported"
            watchConnectionLabel.textColor = .systemRed
            startMonitoringButton.isEnabled = false
            startMonitoringButton.alpha = 0.5
        }
    }
    
    @objc private func startMonitoring() {
        guard WCSession.default.activationState == .activated && WCSession.default.isReachable else {
            showAlert(title: "Error", message: "Apple Watch is not reachable.")
            return
        }
        
        // Send start monitoring message to the watch along with the user ID
        let message: [String: Any] = [
            "command": "startMonitoring",
            "userId": userId
        ]
        
        WCSession.default.sendMessage(message, replyHandler: nil, errorHandler: { error in
            print("Error sending message: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.showAlert(title: "Error", message: "Failed to start monitoring: \(error.localizedDescription)")
            }
        })
        
        // Update UI
        statusLabel.text = "Status: Monitoring"
        statusLabel.textColor = .systemGreen
        startMonitoringButton.isEnabled = false
        startMonitoringButton.alpha = 0.5
        stopMonitoringButton.isEnabled = true
        stopMonitoringButton.alpha = 1.0
        isMonitoring = true
    }
    
    @objc private func stopMonitoring() {
        // Send stop monitoring message to the watch
        let message = ["command": "stopMonitoring"]
        WCSession.default.sendMessage(message, replyHandler: nil, errorHandler: { error in
            print("Error sending message: \(error.localizedDescription)")
        })
        
        // Update UI
        statusLabel.text = "Status: Not Monitoring"
        statusLabel.textColor = .label
        startMonitoringButton.isEnabled = true
        startMonitoringButton.alpha = 1.0
        stopMonitoringButton.isEnabled = false
        stopMonitoringButton.alpha = 0.5
        isMonitoring = false
    }
    
    @objc private func updateHeartRate(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let heartRate = userInfo["heartRate"] as? Double,
              let timestamp = userInfo["timestamp"] as? Double else {
            return
        }
        
        // Update UI
        DispatchQueue.main.async {
            self.heartRateLabel.text = "Heart Rate: \(Int(heartRate)) BPM"
            
            // Update chart data
            self.heartRates.append(heartRate)
            self.timestamps.append(timestamp)
            
            // Keep only last 60 values
            if self.heartRates.count > 60 {
                self.heartRates.removeFirst()
                self.timestamps.removeFirst()
            }
            
            self.updateChart()
        }
    }
    
    @objc private func updateSleepState(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let isREM = userInfo["isREM"] as? Bool else {
            return
        }
        
        DispatchQueue.main.async {
            if isREM {
                self.sleepStateLabel.text = "Sleep State: REM Sleep"
                self.sleepStateLabel.textColor = .systemPurple
            } else {
                self.sleepStateLabel.text = "Sleep State: Non-REM or Awake"
                self.sleepStateLabel.textColor = .systemBlue
            }
        }
    }
    
    @objc private func newREMWindowDetected(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let remWindow = userInfo["remWindow"] as? NightmareDetectionManager.REMSleepWindow else {
            return
        }
        
        // Add to our list of REM windows
        remWindows.append(remWindow)
        
        // Format the time for display
        let startDate = Date(timeIntervalSince1970: remWindow.startTime)
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss"
        let timeString = dateFormatter.string(from: startDate)
        
        // Show a notification if it was a nightmare
        if remWindow.isNightmare {
            DispatchQueue.main.async {
                self.showAlert(
                    title: "Nightmare Detected",
                    message: "A nightmare was detected during REM sleep at \(timeString). Duration: \(Int(remWindow.endTime - remWindow.startTime)) seconds."
                )
            }
        }
    }
    
    @objc private func handleNightmareDetection(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let score = userInfo["score"] as? Double else {
            return
        }
        
        DispatchQueue.main.async {
            // Update UI
            if score > 0.7 {
                self.nightmareRiskLabel.text = "Nightmare Risk: HIGH!"
                self.nightmareRiskLabel.textColor = .systemRed
                
                // Vibrate the phone
                let feedbackGenerator = UINotificationFeedbackGenerator()
                feedbackGenerator.notificationOccurred(.warning)
                
                // Show alert
                self.showAlert(title: "Nightmare Detected", 
                              message: "High probability of nightmare detected. Intervention initiated.")
            } else if score > 0.4 {
                self.nightmareRiskLabel.text = "Nightmare Risk: Medium"
                self.nightmareRiskLabel.textColor = .systemOrange
            } else {
                self.nightmareRiskLabel.text = "Nightmare Risk: Low"
                self.nightmareRiskLabel.textColor = .systemGreen
            }
        }
    }
    
    @objc private func showREMHistory() {
        let historyVC = REMHistoryViewController(remWindows: remWindows)
        navigationController?.pushViewController(historyVC, animated: true)
    }
    
    private func updateChart() {
        var entries = [ChartDataEntry]()
        
        for i in 0..<heartRates.count {
            entries.append(ChartDataEntry(x: Double(i), y: heartRates[i]))
        }
        
        let dataSet = LineChartDataSet(entries: entries, label: "Heart Rate")
        dataSet.drawCirclesEnabled = false
        dataSet.lineWidth = 2
        dataSet.setColor(.systemBlue)
        dataSet.mode = .cubicBezier
        
        let data = LineChartData(dataSet: dataSet)
        heartRateChart.data = data
        heartRateChart.animate(xAxisDuration: 0.5)
    }
    
    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

// A simple view controller to display REM sleep history
class REMHistoryViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {
    private let tableView = UITableView()
    private let remWindows: [NightmareDetectionManager.REMSleepWindow]
    private let dateFormatter = DateFormatter()
    
    init(remWindows: [NightmareDetectionManager.REMSleepWindow]) {
        self.remWindows = remWindows.sorted { $0.startTime > $1.startTime } // Most recent first
        super.init(nibName: nil, bundle: nil)
        dateFormatter.dateFormat = "MMM d, HH:mm:ss"
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        title = "REM Sleep History"
        view.backgroundColor = .systemBackground
        
        tableView.dataSource = self
        tableView.delegate = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "REMCell")
        
        view.addSubview(tableView)
        tableView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor)
        ])
    }
    
    // UITableViewDataSource
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return remWindows.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "REMCell", for: indexPath)
        let window = remWindows[indexPath.row]
        
        let startDate = Date(timeIntervalSince1970: window.startTime)
        let duration = Int(window.endTime - window.startTime)
        
        cell.textLabel?.text = "\(dateFormatter.string(from: startDate)) - Duration: \(duration)s"
        
        // Show nightmare status
        if window.isNightmare {
            cell.textLabel?.textColor = .systemRed
            cell.accessoryType = .detailDisclosureButton
        } else {
            cell.textLabel?.textColor = .label
            cell.accessoryType = .none
        }
        
        return cell
    }
    
    // UITableViewDelegate
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        
        let window = remWindows[indexPath.row]
        let startDate = Date(timeIntervalSince1970: window.startTime)
        let endDate = Date(timeIntervalSince1970: window.endTime)
        
        let alertTitle = window.isNightmare ? "Nightmare Episode" : "REM Sleep Episode"
        let message = """
        Start: \(dateFormatter.string(from: startDate))
        End: \(dateFormatter.string(from: endDate))
        Duration: \(Int(window.endTime - window.startTime)) seconds
        Heart rate data points: \(window.heartRates.count)
        Motion data points: \(window.motionData.count)
        """
        
        let alert = UIAlertController(title: alertTitle, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let watchConnectivityChanged = Notification.Name("WatchConnectivityChanged")
    static let heartRateUpdated = Notification.Name("HeartRateUpdated")
    static let remSleepStateChanged = Notification.Name("REMSleepStateChanged")
    static let newREMWindowDetected = Notification.Name("NewREMWindowDetected")
}
