import Foundation

protocol APIClient: Sendable {
    func request<T: Decodable & Sendable>(
        _ request: URLRequest
    ) async throws -> T
}
