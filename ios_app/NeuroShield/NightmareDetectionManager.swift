// NeuroShield - iOS App
// NightmareDetectionManager.swift

import Foundation
import CoreML
import Accelerate

class NightmareDetectionManager {
    static let shared = NightmareDetectionManager()
    
    // CoreML model
    private var nightmareModel: MLModel?
    
    // User identifier
    private var userId: String = UUID().uuidString
    
    // Data buffers - separate data by REM window
    private var heartRateData: [REMSleepWindow] = []
    private var currentREMWindow: REMSleepWindow?
    
    // Data structures
    struct REMSleepWindow {
        let id: String
        let startTime: Double
        var endTime: Double
        var heartRates: [(time: Double, bpm: Double)] = []
        var motionData: [(time: Double, x: Double, y: Double, z: Double)] = []
        var isNightmare: Bool = false
        var processed: Bool = false
    }
    
    // Constants
    private let windowLength = 120 // 120 seconds as per original model
    private let minimumDataPoints = 30
    private let requiredWindowCount = 5 // Number of data points required to form a valid window
    
    // Status
    private var isREMSleepDetected = false
    private var lastNightmareScore: Double = 0.0
    private var consecutiveLowActivityPeriods = 0
    
    private init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            // Load the CoreML model
            // In production, you would use the compiled model URL
            let modelURL = Bundle.main.url(forResource: "NightmareDetectionModel", withExtension: "mlmodel")!
            nightmareModel = try MLModel(contentsOf: modelURL)
            print("Model loaded successfully")
        } catch {
            print("Failed to load model: \(error.localizedDescription)")
        }
    }
    
    func setUserId(_ id: String) {
        userId = id
    }
    
    func processWatchData(_ data: Data) {
        do {
            // Decode the data from Apple Watch
            if let watchData = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                let timestamp = watchData["timestamp"] as? Double ?? Date().timeIntervalSince1970
                
                // Process heart rate data
                if let heartRate = watchData["heartRate"] as? Double {
                    processHeartRateData(heartRate, timestamp: timestamp)
                }
                
                // Process motion data
                if let accX = watchData["accelerationX"] as? Double, 
                   let accY = watchData["accelerationY"] as? Double, 
                   let accZ = watchData["accelerationZ"] as? Double {
                    processMotionData(accX, accY, accZ, timestamp: timestamp)
                }
                
                // Detect REM sleep based on activity patterns from motion data
                detectREMSleep(timestamp: timestamp)
                
                // If in REM sleep, check for nightmares
                if isREMSleepDetected, let currentWindow = currentREMWindow, 
                   currentWindow.heartRates.count >= minimumDataPoints && 
                   currentWindow.motionData.count >= minimumDataPoints {
                    runNightmareDetection(for: currentWindow)
                }
            }
        } catch {
            print("Error processing watch data: \(error.localizedDescription)")
        }
    }
    
    // Additional methods to process data directly from AppDelegate
    func processHeartRateData(_ bpm: Double, timestamp: Double) {
        if let window = currentREMWindow {
            // Add data to current REM window
            var updatedWindow = window
            updatedWindow.heartRates.append((time: timestamp, bpm: bpm))
            updatedWindow.endTime = timestamp
            currentREMWindow = updatedWindow
        } else if isREMSleepDetected {
            // Create a new REM window if we're in REM sleep
            currentREMWindow = REMSleepWindow(
                id: userId,
                startTime: timestamp,
                endTime: timestamp,
                heartRates: [(time: timestamp, bpm: bpm)],
                motionData: []
            )
        }
    }
    
    func processMotionData(_ x: Double, _ y: Double, _ z: Double, timestamp: Double) {
        if let window = currentREMWindow {
            // Add data to current REM window
            var updatedWindow = window
            updatedWindow.motionData.append((time: timestamp, x: x, y: y, z: z))
            updatedWindow.endTime = timestamp
            currentREMWindow = updatedWindow
        } else if isREMSleepDetected {
            // Create a new REM window if we're in REM sleep but don't have a window yet
            currentREMWindow = REMSleepWindow(
                id: userId,
                startTime: timestamp,
                endTime: timestamp,
                heartRates: [],
                motionData: [(time: timestamp, x: x, y: y, z: z)]
            )
        }
    }
    
    func updateREMSleepState(_ isREM: Bool) {
        let wasInREM = isREMSleepDetected
        isREMSleepDetected = isREM
        
        // If we just entered REM sleep, create a new window
        if isREMSleepDetected && !wasInREM {
            let timestamp = Date().timeIntervalSince1970
            currentREMWindow = REMSleepWindow(
                id: userId,
                startTime: timestamp,
                endTime: timestamp,
                heartRates: [],
                motionData: []
            )
        }
        
        // If we just exited REM sleep, finish the current window
        if !isREMSleepDetected && wasInREM {
            finishCurrentREMWindow()
        }
    }
    
    func setREMStartTime(_ startTime: Double) {
        if var window = currentREMWindow, window.startTime > startTime {
            window.startTime = startTime
            currentREMWindow = window
        }
    }
    
    private func detectREMSleep(timestamp: Double) {
        // This is a simple approximation - in reality, you would use a more sophisticated
        // algorithm based on heart rate variability, respiration, and other signals
        
        // If we have a current window, check if it's been too long since last update
        if let window = currentREMWindow, timestamp - window.endTime > 60 {
            // If more than 60 seconds have passed, close this REM window
            finishCurrentREMWindow()
            isREMSleepDetected = false
            consecutiveLowActivityPeriods = 0
            return
        }
        
        // Check recent motion activity (assuming we have recent data)
        if let window = currentREMWindow, window.motionData.count >= 10 {
            // Calculate average motion magnitude over recent data points
            let recentMotion = window.motionData.suffix(10)
            let avgMagnitude = recentMotion.reduce(0.0) { sum, motion in
                let magnitude = sqrt(motion.x * motion.x + motion.y * motion.y + motion.z * motion.z)
                return sum + magnitude
            } / Double(recentMotion.count)
            
            // Check if heart rate is in typical REM range (60-100 BPM) and motion is low
            if window.heartRates.count >= 5 {
                let recentHeartRates = window.heartRates.suffix(5)
                let avgHeartRate = recentHeartRates.reduce(0.0) { $0 + $1.bpm } / Double(recentHeartRates.count)
                
                // Low activity + heart rate in REM range suggests REM sleep
                if avgMagnitude < 0.2 && avgHeartRate >= 60 && avgHeartRate <= 100 {
                    consecutiveLowActivityPeriods += 1
                } else {
                    consecutiveLowActivityPeriods = max(0, consecutiveLowActivityPeriods - 1)
                }
            }
        }
        
        // Need several consecutive periods of low activity to confirm REM
        let wasInREM = isREMSleepDetected
        isREMSleepDetected = consecutiveLowActivityPeriods >= 5
        
        // If we just entered REM sleep, create a new window
        if isREMSleepDetected && !wasInREM {
            currentREMWindow = REMSleepWindow(
                id: userId,
                startTime: timestamp,
                endTime: timestamp,
                heartRates: [],
                motionData: []
            )
        }
        
        // If we just exited REM sleep, finish the current window
        if !isREMSleepDetected && wasInREM {
            finishCurrentREMWindow()
        }
    }
    
    private func finishCurrentREMWindow() {
        if var window = currentREMWindow, 
           window.endTime - window.startTime >= 60, // At least 1 minute of data
           window.heartRates.count >= minimumDataPoints,
           window.motionData.count >= minimumDataPoints {
            
            // Run nightmare detection one last time
            runNightmareDetection(for: window)
            
            // Mark as processed and save
            window.processed = true
            heartRateData.append(window)
            
            // Clear current window
            currentREMWindow = nil
        } else {
            // Not enough data, discard
            currentREMWindow = nil
        }
    }
    
    private func runNightmareDetection(for window: REMSleepWindow) {
        // Extract features from the window data
        let features = extractFeatures(from: window)
        guard !features.isEmpty else { return }
        
        do {
            // Create an MLMultiArray with the right shape for the model input
            let featureArray = try MLMultiArray(shape: [1, NSNumber(value: features.count), NSNumber(value: features[0].count)], dataType: .double)
            
            // Fill the array with our features
            for (i, featureVector) in features.enumerated() {
                for (j, value) in featureVector.enumerated() {
                    featureArray[[0, NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: value)
                }
            }
            
            // Create length tensor (sequence length)
            let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
            lengthArray[0] = NSNumber(value: features.count)
            
            // Create the model input
            let input = NightmareDetectionModelInput(
                features: featureArray,
                lengths: lengthArray
            )
            
            // Run inference
            if let model = nightmareModel {
                let prediction = try model.prediction(from: input)
                
                // Get prediction results
                if let outputFeatures = prediction.featureValue(for: "output"),
                   let probabilities = outputFeatures.multiArrayValue {
                    
                    // Get nightmare probability (class 1)
                    let nightmareScore = probabilities[1].doubleValue
                    lastNightmareScore = nightmareScore
                    
                    // If probability is high, trigger an alert or intervention
                    if nightmareScore > 0.7 {
                        // Update current window with nightmare status
                        if var currentWindow = currentREMWindow, !currentWindow.processed {
                            currentWindow.isNightmare = true
                            currentREMWindow = currentWindow
                        }
                        
                        // Trigger intervention
                        triggerIntervention(nightmareScore: nightmareScore)
                    }
                }
            }
        } catch {
            print("Prediction error: \(error.localizedDescription)")
        }
    }
    
    private func extractFeatures(from window: REMSleepWindow) -> [[Double]] {
        // Create an empty array to hold the feature vectors
        var featuresList: [[Double]] = []
        
        // Make sure we have enough data
        guard window.heartRates.count >= requiredWindowCount, 
              window.motionData.count >= requiredWindowCount else {
            return []
        }
        
        // Create aligned time series by interpolating or sampling
        let (heartRateSeries, motionSeries) = alignTimeSeriesData(
            heartRates: window.heartRates,
            motionData: window.motionData
        )
        
        // Make sure we have enough aligned data
        guard heartRateSeries.count >= requiredWindowCount else {
            return []
        }
        
        // Calculate statistics over the entire window
        let bpmValues = heartRateSeries.map { $0.bpm }
        let meanBPM = bpmValues.reduce(0, +) / Double(bpmValues.count)
        let stdBPM = standardDeviation(bpmValues)
        let maxBPM = bpmValues.max() ?? 0
        let minBPM = bpmValues.min() ?? 0
        
        // Calculate BPM gradient
        let bpmGradient = gradient(bpmValues)
        let meanGradientBPM = bpmGradient.reduce(0, +) / Double(bpmGradient.count)
        
        // Prepare feature vectors for each time step
        for i in 0..<heartRateSeries.count {
            // Get aligned data points
            let heartRate = heartRateSeries[i].bpm
            let motion = motionSeries[i]
            
            // Acceleration data
            let accX = motion.x
            let accY = motion.y
            let accZ = motion.z
            
            // Magnitude and energy
            let magnitude = sqrt(accX*accX + accY*accY + accZ*accZ)
            let energy = accX*accX + accY*accY + accZ*accZ
            
            // Jerk (derivative of acceleration)
            let jerkX = i > 0 ? accX - motionSeries[i-1].x : 0
            let jerkY = i > 0 ? accY - motionSeries[i-1].y : 0
            let jerkZ = i > 0 ? accZ - motionSeries[i-1].z : 0
            
            // Magnitude gradient
            let prevMagnitude = i > 0 ? sqrt(
                motionSeries[i-1].x * motionSeries[i-1].x +
                motionSeries[i-1].y * motionSeries[i-1].y +
                motionSeries[i-1].z * motionSeries[i-1].z
            ) : 0
            let magnitudeGradient = i > 0 ? magnitude - prevMagnitude : 0
            
            // Create feature vector similar to the Python model
            let featureVector: [Double] = [
                heartRate,                           // Current BPM
                accX, accY, accZ,                    // Acceleration XYZ
                meanBPM, stdBPM, maxBPM, minBPM,     // Heart rate statistics
                meanGradientBPM,                     // Mean gradient BPM
                i < bpmGradient.count ? bpmGradient[i] : 0,  // Current gradient BPM
                magnitude,                           // Acceleration magnitude
                magnitudeGradient,                   // Magnitude gradient
                energy,                              // Motion energy
                jerkX, jerkY, jerkZ                  // Jerk XYZ
            ]
            
            featuresList.append(featureVector)
        }
        
        return featuresList
    }
    
    private func alignTimeSeriesData(
        heartRates: [(time: Double, bpm: Double)],
        motionData: [(time: Double, x: Double, y: Double, z: Double)]
    ) -> ([(time: Double, bpm: Double)], [(time: Double, x: Double, y: Double, z: Double)]) {
        // Sort by time to ensure chronological order
        let sortedHeartRates = heartRates.sorted { $0.time < $1.time }
        let sortedMotion = motionData.sorted { $0.time < $1.time }
        
        // Find the common time range
        let heartRateStartTime = sortedHeartRates.first?.time ?? 0
        let heartRateEndTime = sortedHeartRates.last?.time ?? 0
        let motionStartTime = sortedMotion.first?.time ?? 0
        let motionEndTime = sortedMotion.last?.time ?? 0
        
        let startTime = max(heartRateStartTime, motionStartTime)
        let endTime = min(heartRateEndTime, motionEndTime)
        
        // Ensure we have a valid time range
        guard startTime < endTime else {
            return ([], [])
        }
        
        // Filter data to the common time range
        let filteredHeartRates = sortedHeartRates.filter { $0.time >= startTime && $0.time <= endTime }
        let filteredMotion = sortedMotion.filter { $0.time >= startTime && $0.time <= endTime }
        
        // Determine sampling points - we'll use heart rate timestamps as reference
        // and interpolate motion data to those times
        let samplingPoints = filteredHeartRates
        
        // Interpolate motion data to heart rate timestamps
        var alignedMotion: [(time: Double, x: Double, y: Double, z: Double)] = []
        
        for point in samplingPoints {
            // Find motion data points that bracket this timestamp
            let beforeMotion = filteredMotion.last { $0.time <= point.time }
            let afterMotion = filteredMotion.first { $0.time > point.time }
            
            if let before = beforeMotion, let after = afterMotion {
                // Interpolate using linear interpolation
                let timeRatio = (point.time - before.time) / (after.time - before.time)
                let interpolatedX = before.x + timeRatio * (after.x - before.x)
                let interpolatedY = before.y + timeRatio * (after.y - before.y)
                let interpolatedZ = before.z + timeRatio * (after.z - before.z)
                
                alignedMotion.append((time: point.time, x: interpolatedX, y: interpolatedY, z: interpolatedZ))
            } else if let before = beforeMotion {
                // Use the last available motion data
                alignedMotion.append(before)
            } else if let after = afterMotion {
                // Use the first available motion data
                alignedMotion.append(after)
            }
        }
        
        return (samplingPoints, alignedMotion)
    }
    
    private func gradient(_ values: [Double]) -> [Double] {
        guard values.count > 1 else { return [0] }
        
        var result = [Double](repeating: 0, count: values.count)
        for i in 1..<values.count-1 {
            result[i] = (values[i+1] - values[i-1]) / 2.0
        }
        result[0] = values[1] - values[0]
        result[values.count-1] = values[values.count-1] - values[values.count-2]
        
        return result
    }
    
    private func standardDeviation(_ values: [Double]) -> Double {
        let count = Double(values.count)
        guard count > 1 else { return 0 }
        
        let mean = values.reduce(0, +) / count
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / (count - 1)
        return sqrt(variance)
    }
    
    private func triggerIntervention(nightmareScore: Double) {
        // Send a message to the Watch app to trigger vibration or sound
        if let session = (UIApplication.shared.delegate as? AppDelegate)?.session, session.isReachable {
            // Trigger intervention based on score
            let message = ["action": "triggerIntervention", "score": nightmareScore]
            try? session.updateApplicationContext(message)
            
            // Also notify the user on the phone
            NotificationCenter.default.post(name: .nightmareDetected, object: nil, userInfo: ["score": nightmareScore])
        }
    }
}

// MARK: - Model Input/Output Extensions
extension Notification.Name {
    static let nightmareDetected = Notification.Name("NightmareDetected")
}

// This would be generated from converting the PyTorch model to CoreML
struct NightmareDetectionModelInput: MLFeatureProvider {
    let features: MLMultiArray
    let lengths: MLMultiArray
    
    var featureNames: Set<String> {
        return ["features", "lengths"]
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "features" {
            return MLFeatureValue(multiArray: features)
        } else if featureName == "lengths" {
            return MLFeatureValue(multiArray: lengths)
        }
        return nil
    }
}
