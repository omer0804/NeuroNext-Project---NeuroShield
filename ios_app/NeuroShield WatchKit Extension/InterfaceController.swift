// NeuroShield - WatchKit App
// InterfaceController.swift

import WatchKit
import Foundation
import HealthKit
import CoreMotion
import WatchConnectivity

class InterfaceController: WKInterfaceController {
    
    // UI elements
    @IBOutlet weak var statusLabel: WKInterfaceLabel!
    @IBOutlet weak var heartRateLabel: WKInterfaceLabel!
    @IBOutlet weak var startButton: WKInterfaceButton!
    @IBOutlet weak var stopButton: WKInterfaceButton!
    
    // Health and motion managers
    private let healthStore = HKHealthStore()
    private let motionManager = CMMotionManager()
    
    // WatchConnectivity
    private var session: WCSession?
    
    // User ID - this would be set during app setup
    private var userId: String = "user_\(Int(Date().timeIntervalSince1970))"
    
    // Monitoring state
    private var isMonitoring = false
    private var heartRateQuery: HKQuery?
    
    // Data buffers for batching
    private var heartRateBuffer: [(time: Double, bpm: Double)] = []
    private var motionBuffer: [(time: Double, x: Double, y: Double, z: Double)] = []
    private let maxBufferSize = 20 // Send data in batches
    
    // Data sampling
    private let heartRateSamplingInterval: TimeInterval = 5.0 // seconds
    private let motionSamplingInterval: TimeInterval = 0.1    // seconds
    
    // REM detection
    private var potentiallyInREM = false
    private var remStartTime: TimeInterval = 0
    
    override func awake(withContext context: Any?) {
        super.awake(withContext: context)
        setupWatchConnectivity()
        setupHealthKit()
    }
    
    override func willActivate() {
        super.willActivate()
        updateUI()
    }
    
    override func didDeactivate() {
        super.didDeactivate()
    }
    
    // MARK: - Setup
    
    private func setupWatchConnectivity() {
        if WCSession.isSupported() {
            session = WCSession.default
            session?.delegate = self
            session?.activate()
        }
    }
    
    private func setupHealthKit() {
        // Request authorization for heart rate and activity
        let typesToRead: Set<HKObjectType> = [
            HKQuantityType.quantityType(forIdentifier: .heartRate)!,
            HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!,
            HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            if !success {
                print("HealthKit authorization failed: \(String(describing: error?.localizedDescription))")
            }
        }
    }
    
    // MARK: - UI Actions
    
    @IBAction func startMonitoring() {
        startHealthMonitoring()
        startMotionMonitoring()
        startSleepDetection()
        
        isMonitoring = true
        updateUI()
    }
    
    @IBAction func stopMonitoring() {
        stopHealthMonitoring()
        stopMotionMonitoring()
        
        // Send any remaining data in buffers
        sendBufferedData()
        
        isMonitoring = false
        updateUI()
    }
    
    // MARK: - Monitoring Methods
    
    private func startHealthMonitoring() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        
        // Setup streaming query for heart rate
        let predicate = HKQuery.predicateForSamples(withStart: Date(), end: nil, options: .strictStartDate)
        
        heartRateQuery = HKAnchoredObjectQuery(type: heartRateType, predicate: predicate, anchor: nil, limit: HKObjectQueryNoLimit) { query, samples, deletedObjects, anchor, error in
            self.processHeartRateSamples(samples)
        }
        
        heartRateQuery?.updateHandler = { query, samples, deletedObjects, anchor, error in
            self.processHeartRateSamples(samples)
        }
        
        healthStore.execute(heartRateQuery!)
    }
    
    private func stopHealthMonitoring() {
        if let query = heartRateQuery {
            healthStore.stop(query)
            heartRateQuery = nil
        }
    }
    
    private func startMotionMonitoring() {
        if motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = motionSamplingInterval
            
            motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, error in
                guard let self = self, let accelerationData = data else { return }
                
                // Process and buffer acceleration data
                self.processAccelerationData(accelerationData)
            }
        }
    }
    
    private func stopMotionMonitoring() {
        if motionManager.isAccelerometerActive {
            motionManager.stopAccelerometerUpdates()
        }
    }
    
    private func startSleepDetection() {
        // In a real app, we would use a combination of:
        // 1. HealthKit sleep analysis (if available)
        // 2. Motion and heart rate patterns
        // 3. Time of day
        // 4. User input about when they went to bed
        
        // For demonstration, we'll just periodically check for REM-like patterns
        let timer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            self?.checkForREMSleep()
        }
        RunLoop.current.add(timer, forMode: .common)
    }
    
    // MARK: - Data Processing
    
    private func processHeartRateSamples(_ samples: [HKSample]?) {
        guard let samples = samples as? [HKQuantitySample] else { return }
        
        for sample in samples {
            let heartRate = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            let timestamp = sample.startDate.timeIntervalSince1970
            
            // Update UI
            DispatchQueue.main.async {
                self.heartRateLabel.setText("❤️ \(Int(heartRate)) BPM")
            }
            
            // Buffer heart rate data
            self.heartRateBuffer.append((time: timestamp, bpm: heartRate))
            
            // If buffer reaches threshold, send to phone
            if self.heartRateBuffer.count >= self.maxBufferSize {
                self.sendBufferedData()
            }
        }
    }
    
    private func processAccelerationData(_ accelerometerData: CMAccelerometerData) {
        let timestamp = Date().timeIntervalSince1970
        let x = accelerometerData.acceleration.x
        let y = accelerometerData.acceleration.y
        let z = accelerometerData.acceleration.z
        
        // Buffer motion data
        motionBuffer.append((time: timestamp, x: x, y: y, z: z))
        
        // Check if buffer is full
        if motionBuffer.count >= maxBufferSize {
            sendBufferedData()
        }
    }
    
    private func sendBufferedData() {
        guard let session = session, session.activationState == .activated, isMonitoring else {
            return
        }
        
        // Prepare heart rate data in the format expected by the model
        let heartRateData: [[String: Any]] = heartRateBuffer.map { 
            ["time": $0.time, "bpm": $0.bpm, "id": userId] 
        }
        
        // Prepare motion data in the format expected by the model
        let motionData: [[String: Any]] = motionBuffer.map { 
            [
                "time": $0.time, 
                "acceleration_x": $0.x, 
                "acceleration_y": $0.y, 
                "acceleration_z": $0.z,
                "id": userId
            ] 
        }
        
        // Include REM sleep status
        let isREM = potentiallyInREM
        let remInfo: [String: Any] = [
            "isREM": isREM,
            "remStartTime": remStartTime,
            "remDuration": isREM ? Date().timeIntervalSince1970 - remStartTime : 0
        ]
        
        // Combine all data
        let dataPackage: [String: Any] = [
            "userId": userId,
            "heartRateData": heartRateData,
            "motionData": motionData,
            "remInfo": remInfo,
            "timestamp": Date().timeIntervalSince1970
        ]
        
        do {
            let messageData = try JSONSerialization.data(withJSONObject: dataPackage)
            session.sendMessageData(messageData, replyHandler: nil) { error in
                print("Error sending data package: \(error.localizedDescription)")
            }
            
            // Clear buffers after sending
            heartRateBuffer.removeAll()
            motionBuffer.removeAll()
        } catch {
            print("Failed to serialize data package: \(error.localizedDescription)")
        }
    }
    
    // MARK: - REM Sleep Detection
    
    private func checkForREMSleep() {
        // This is a simplified approach - a real app would use more sophisticated algorithms
        
        // Criteria for potential REM sleep:
        // 1. Low physical activity (from accelerometer)
        // 2. Heart rate variations typical of REM
        // 3. Time of day is typically overnight
        
        let calendar = Calendar.current
        let hourOfDay = calendar.component(.hour, from: Date())
        let isNightTime = hourOfDay >= 22 || hourOfDay <= 6
        
        // Check recent motion activity
        let recentMotionActivity = calculateRecentMotionActivity()
        
        // Check heart rate pattern
        let hasREMHeartRatePattern = checkForREMHeartRatePattern()
        
        // Determine if in REM sleep
        let wasInREM = potentiallyInREM
        potentiallyInREM = isNightTime && recentMotionActivity < 0.2 && hasREMHeartRatePattern
        
        // If just entered REM, note the start time
        if potentiallyInREM && !wasInREM {
            remStartTime = Date().timeIntervalSince1970
            
            // Update UI to indicate REM sleep detected
            DispatchQueue.main.async {
                self.statusLabel.setText("REM Sleep Detected")
            }
        } 
        // If just exited REM, send any remaining data
        else if !potentiallyInREM && wasInREM {
            sendBufferedData()
            
            DispatchQueue.main.async {
                self.statusLabel.setText("Monitoring")
            }
        }
    }
    
    private func calculateRecentMotionActivity() -> Double {
        guard !motionBuffer.isEmpty else { return 0 }
        
        // Calculate average magnitude of acceleration
        return motionBuffer.reduce(0.0) { sum, motion in
            let magnitude = sqrt(motion.x * motion.x + motion.y * motion.y + motion.z * motion.z)
            return sum + magnitude
        } / Double(motionBuffer.count)
    }
    
    private func checkForREMHeartRatePattern() -> Bool {
        guard heartRateBuffer.count >= 10 else { return false }
        
        // Get recent heart rates
        let recentHeartRates = heartRateBuffer.suffix(10).map { $0.bpm }
        
        // Calculate heart rate variability (simple standard deviation)
        let mean = recentHeartRates.reduce(0, +) / Double(recentHeartRates.count)
        let variance = recentHeartRates.reduce(0) { sum, rate in
            let diff = rate - mean
            return sum + (diff * diff)
        } / Double(recentHeartRates.count)
        let stdDev = sqrt(variance)
        
        // Look for moderate variability and rates typical of REM
        return stdDev > 3 && stdDev < 10 && mean > 55 && mean < 100
    }
    
    // MARK: - Helper Methods
    
    private func updateUI() {
        DispatchQueue.main.async {
            if self.isMonitoring {
                self.statusLabel.setText("Monitoring")
                self.startButton.setHidden(true)
                self.stopButton.setHidden(false)
            } else {
                self.statusLabel.setText("Not Monitoring")
                self.startButton.setHidden(false)
                self.stopButton.setHidden(true)
                self.heartRateLabel.setText("--")
            }
        }
    }
    
    // MARK: - Handle Interventions
    
    private func handleNightmareIntervention() {
        // Vibrate the watch
        WKInterfaceDevice.current().play(.notification)
        
        // Flash the screen
        DispatchQueue.main.async {
            self.statusLabel.setText("NIGHTMARE DETECTED!")
            
            // Reset status after 5 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                if self.isMonitoring {
                    self.statusLabel.setText(self.potentiallyInREM ? "REM Sleep Detected" : "Monitoring")
                }
            }
        }
    }
}

// MARK: - WCSessionDelegate

extension InterfaceController: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        print("WatchKit: Session activation completed")
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        DispatchQueue.main.async {
            // Handle commands from iPhone app
            if let command = message["command"] as? String {
                switch command {
                case "startMonitoring":
                    self.startMonitoring()
                case "stopMonitoring":
                    self.stopMonitoring()
                case "setUserId":
                    if let userId = message["userId"] as? String {
                        self.userId = userId
                    }
                default:
                    break
                }
            }
            
            // Handle intervention triggers
            if let action = message["action"] as? String, action == "triggerIntervention" {
                self.handleNightmareIntervention()
            }
        }
    }
}
